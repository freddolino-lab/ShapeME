///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
// todo: job monitoring in dashboard
// todo: job reports
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////


//use std::error::Error;
//use std::io::ErrorKind;
use std::fmt;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::io::Cursor;
use image::io::Reader as ImageReader;
use glob::glob;
use base64::{Engine as _, engine::{self, general_purpose}, alphabet};

use crypto::digest::Digest;
use crypto::sha3::Sha3;

//use rocket::{Rocket, Build};
use rocket::tokio::process::{Command, Child};
use rocket::fairing::AdHoc;
use rocket::serde::{Serialize, Deserialize, json::Json};
use rocket::response::{Debug, content::RawHtml};//, status::Created};
use rocket::fs::relative;
use rocket::form::{Form, Contextual, Context, DataField, FromFormField};
use rocket::FromForm;
use rocket_dyn_templates::Template;
use rocket::http::{Status, CookieJar, Cookie};
use rocket::data::ToByteUnit;
use rocket::tokio;

use rocket_sync_db_pools::{database, rusqlite};
use self::rusqlite::params;

use serde_json;
use rand::{self, Rng};

const JOB_ID_LENGTH: usize = 12;

#[database("shapeme")]
struct Db(rusqlite::Connection);

#[derive(Debug,Clone)]
struct NoUserError;

#[derive(Debug,Clone)]
struct NoLoggedInUserError;

impl fmt::Display for NoLoggedInUserError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "No user logged in. Redirecting to login page.")
    }
}

impl Error for NoLoggedInUserError {}

type DbResult<T, E = Debug<rusqlite::Error>> = std::result::Result<T, E>;

fn hash_password(password: &str) -> String {
    let mut hasher = Sha3::sha3_256();
    hasher.input_str(password);
    hasher.result_str()
}

#[derive(Debug, FromForm)]
#[allow(dead_code)]
struct CreateUser<'a> {
    #[field(validate = len(1..))]
    first: String,
    #[field(validate = len(1..))]
    last: String,
    password: CreatePassword<'a>,
    #[field(validate = contains('@').or_else(msg!("invalid email address")))]
    pub email: String,
    #[field(validate = len(1..))]
    lab_head: String,
}

#[derive(Debug, FromForm)]
struct CreatePassword<'a> {
    #[field(validate = len(6..))]
    #[field(validate = eq(self.second))]
    first: &'a str,
    #[field(validate = eq(self.first))]
    second: &'a str,
}

//#[derive(Debug, Clone, Deserialize, Serialize, FromForm)]
#[derive(Debug)]
//#[serde(crate = "rocket::serde")]
struct User {
    uid: Option<i32>,
    first: Option<String>,
    last: Option<String>,
    email: String,
    password_hash: Option<String>,
    lab_head: Option<String>,
}

//#[derive(Debug, Serialize, Deserialize)]
//struct UserContext {
//    uid: &i32,
//}

#[derive(Debug, Clone, Deserialize, Serialize, FromForm)]
#[serde(crate = "rocket::serde")]
struct LoginUser {
    #[field(validate = contains('@').or_else(msg!("invalid email address")))]
    email: String,
    #[field(validate = len(6..))]
    password: String,
}

impl User {
    
    async fn check_user(db: &Db, email: String) -> DbResult<User> {
        let user = db.run(move |conn| {
            conn.query_row(
                "SELECT uid, first, last, email, password, lab_head FROM users WHERE email = ?1",
                params![email],
                |r| Ok(User {
                    uid: Some(r.get(0)?),
                    first: Some(r.get(1)?),
                    last: Some(r.get(2)?),
                    email: r.get(3)?,
                    password_hash: r.get(4)?,
                    lab_head: Some(r.get(5)?),
                }))
        }).await?;

        Ok(user)
    }

    fn check_auth(&self, db: &Db, password: &str) -> bool {
        let hash = hash_password(password);
        self.password_hash == Some(hash)
    }

    async fn from_uid(db: &Db, uid: i32) -> DbResult<User> {

        let user = db.run(move |conn| {
            conn.query_row(
                "SELECT uid, first, last, email FROM users WHERE uid = ?1",
                params![uid],
                |r| Ok(User {
                    uid: Some(r.get(0)?),
                    first: Some(r.get(1)?),
                    last: Some(r.get(2)?),
                    email: r.get(3)?,
                    password_hash: None,
                }))
        }).await?;
        Ok(user)
    }

    fn from_create_user(create_user: &CreateUser) -> User {
        let first = create_user.first.clone();
        let last = create_user.last.clone();
        let email = create_user.email.clone();
        let pass = String::from(create_user.password.first);
        let hash = hash_password(&pass);
        let lab_head = create_user.lab_head.clone();
        User{
            uid: None,
            first: Some(first),
            last: Some(last),
            email: email,
            password_hash: Some(hash),
            lab_head: Some(lab_head),
        }
    }

    /// Inserts this user into sqlite database
    async fn insert_into_db(&self, db: &Db) -> DbResult<i32> {
        let first = self.first.as_ref().unwrap().clone();
        let last = self.last.as_ref().unwrap().clone();
        let email = String::from(&self.email);
        let pass_hash = self.password_hash.as_ref().unwrap().clone();
        let lab_head = self.lab_head.as_ref().unwrap().clone();
        //let pass = String::from();
        db.run(move |conn| {
            conn.execute(
                "INSERT INTO users (first, last, email, password, lab_head)
                VALUES (?1, ?2, ?3, ?4, ?5)",
                params![first, last, email, pass_hash, lab_head])
        }).await?;
        let uid: i32 = db.run(move |conn| {
            conn.last_insert_rowid()
        }).await.try_into().unwrap();
        Ok(uid)
    }

    /// Gets job corresponding to the given id
    async fn fetch_job(&self, db: &Db, job_id: &str) -> Result<JobContext, Box<dyn Error>> {
        let uid = self.uid.clone();
        let job_id = job_id.to_string();
        let job_row: JobRow = db.run(move |conn| {
            conn.query_row(
                "SELECT id, name, args, version, uid
                FROM jobs WHERE uid = ?1 and id = ?2",
                params![uid, job_id],
                |row| Ok(JobRow {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    args: row.get(2)?,
                    version: row.get(3)?,
                    uid: row.get(4)?,
            }))
        }).await?;
        //////////////////////////////////////////////////////
        // need to appropriately handle queryreturnednorows error
        //////////////////////////////////////////////////////
        //rusqlite::Error::QueryReturnedNoRows => {
        //    println!("No such user:\n{:?}", &form.context);
        //    Template::render("no_such_user", &form.context)
        //}

        JobContext::from_jobrow(job_row)
    }

    /// Gets every job from db for this user.
    async fn fetch_all_jobs(&self, db: &Db) -> DbResult<Jobs> {
        let jobs_result: Jobs = if let Some(uid) = self.uid {
            db.run(move |conn| {
                let mut stmt = conn
                    .prepare("SELECT id, name, args, version, uid FROM jobs WHERE uid = ?1")
                    .expect("Unable to prepare statement");
                let job_rows_iter = stmt.query_map(params![uid], |row| {
                    Ok(JobRow {
                        id: row.get(0)?,
                        name: row.get(1)?,
                        args: row.get(2)?,
                        version: row.get(3)?,
                        uid: row.get(4)?,
                    })
                }).expect("Db query failed");
                let mut jobs: Vec<JobContext> = Vec::new();
                for job_row_result in job_rows_iter {
                    let job_row = job_row_result.expect("unable to get jobrow");
                    let job_context = JobContext::from_jobrow(job_row)
                        .expect("unable to make jobcontext");
                    jobs.push(job_context);
                }
                Jobs{jobs}
            }).await
        } else {
            let jobs: Vec<JobContext> = Vec::new();
            Jobs{jobs}
        };
        Ok(jobs_result)
    }
}

#[get("/logout")]
fn logout(mut cookies: &CookieJar<'_>) -> Template {
    cookies.remove_private(Cookie::named("uid"));
    Template::render("index", &Context::default())
}

#[get("/")]
fn login_form() -> Template {
    Template::render("index", &Context::default())
}

#[get("/dashboard")]
async fn dashboard(
        db: Db,
        mut cookies: &CookieJar<'_>
) -> Template {
    let maybe_cookie = cookies.get_private("uid");
    // if a user is logged in, serve their dashboard, else redirect to index
    let template = if let Some(cookie) = maybe_cookie {
        let uid = cookie.value().to_string().parse().unwrap();
        let user = User::from_uid(&db, uid).await.unwrap();
        let user_jobs = user.fetch_all_jobs(&db).await.unwrap();
        Template::render("dashboard", &user_jobs)
    } else {
        println!("No user logged in, redirecting to login page.");
        Template::render("index", &Context::default())
    };
    template
}


#[post("/", data = "<form>")]
async fn login(
        db: Db,
        form: Form<Contextual<'_, LoginUser>>,
        mut cookies: &CookieJar<'_>,
) -> (Status, Template) {

    println!("form:\n{:?}", form);
    let template = match form.value {
        Some(ref user) => {
            // check if user exists, if not, serve the create account page
            let email = String::from(&user.email);
            let user_info_result = User::check_user(&db, email).await;
            
            let template = match user_info_result {
                Ok(user_info) => {
                    let uid = format!("{}", &user_info.uid.unwrap());
                    let authorized = user_info.check_auth(&db, &user.password);
                    if authorized {
                        println!("User {:?} authorized", &user.email);
                        cookies.add_private(
                            Cookie::new("uid", uid)
                        );
                        let user_jobs = user_info.fetch_all_jobs(&db).await.unwrap();
                        //println!("user jobs context{:?}", user_jobs);
                        Template::render("dashboard", &user_jobs)
                    } else {
                        println!("User {:?} not authorized", &user.email);
                        Template::render("no_auth", &form.context)
                    }
                },
                Err(error) => match error.0 {
                    rusqlite::Error::QueryReturnedNoRows => {
                        println!("No such user:\n{:?}", &form.context);
                        Template::render("no_such_user", &form.context)
                    },
                    other_error => {
                        Template::render("index", &Context::default())
                    }
                },
            };
            template
        }
        None => {
            println!("None returned on login!!");
            Template::render("index", &form.context)
        }
    };
    (form.context.status(), template)
}

#[get("/create_account")]
fn create_account_form() -> Template {
    Template::render("create_account", &Context::default())
}

#[post("/create_account", data = "<form>")]
async fn create_account(
        db: Db,
        form: Form<Contextual<'_, CreateUser<'_>>>,
        mut cookies: &CookieJar<'_>,
) -> (Status, Template) {
    println!("form:\n{:?}", form);
    let template = match form.value {
        Some(ref create_user) => {
            let user = User::from_create_user(create_user);
            let insert_result = user.insert_into_db(&db).await;

            let template = match insert_result {
                Ok(uid) => {
                    println!("User {} created.", &user.email);
                    cookies.add_private(Cookie::new("uid", uid.to_string()));
                    let user = User::from_uid(&db, uid).await.unwrap();
                    let user_jobs = user.fetch_all_jobs(&db).await.unwrap();
                    Template::render("dashboard", &user_jobs)
                },
                Err(error) => match error.0 {
                    rusqlite::Error::SqliteFailure(err, Some(msg)) => {
                        println!("Error in creating user {}.\n{:?}\n{:?}", &user.email, &err, &msg);
                        Template::render("account_exists", &Context::default())
                    },
                    other_error => {
                        println!("Error in creating user {}.\n{:?}", &user.email, other_error);
                        Template::render("create_account", &Context::default())
                    }
                },
            };
            template
        }
        None => {
            Template::render("create_account", &form.context)
        }
    };
 
    (form.context.status(), template)
}

#[derive(Debug, Serialize, Deserialize)]
pub enum JobStatus {
    Running,
    Queued,
    FinishedWithMotifs,
    FinishedNoMotif,
    FinishedError,
    DoesNotExist,
}

#[derive(Debug, FromForm)]
#[allow(dead_code)]
pub struct Submission {
    #[field(validate = len(1..))]
    name: String,
    //#[field(validate = ext(ContentType::Fasta))]
    fa_file: File,
    //#[field(validate = ext(ContentType::Scores))]
    score_file: File,
}

#[derive(Debug, FromForm)]
#[allow(dead_code)]
pub struct Submit {
    //pub account: Account,
    pub submission: Submission,
    pub cfg: Cfg,
}

impl Submit {
    async fn insert_into_db(
            &self,
            db: &Db,
            uid: &i32,
            job_id: &str,
    ) -> Result<(), Box<dyn Error>> {
        let job_name = self.submission.name.clone();
        let arg_string = self.cfg.build_arg_string()?;
        let id = String::from(job_id);
        let user_id = uid.clone();
        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////
        let version = "0.1.0";
        db.run(move |conn| {
            conn.execute(
                "INSERT INTO jobs (id, name, args, version, uid)
                VALUES (?1, ?2, ?3, ?4, ?5)",
                params![id, job_name, arg_string, version, user_id])
        }).await?;
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct JobContext {
    #[serde(skip_deserializing, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub path: PathBuf,
    fa_path: PathBuf,
    score_path: PathBuf,
    pub status: JobStatus,
    #[serde(skip_deserializing, skip_serializing_if = "Option::is_none")]
    job_row: Option<JobRow>,
}

#[derive(Debug, Serialize)]
struct JobRow {
    id: String,
    name: String,
    args: String,
    version: String,
    uid: i32,
}

#[derive(Debug, Serialize)]
struct Jobs { jobs: Vec<JobContext> }

impl JobContext {

    pub fn to_json(&self, fname: &PathBuf) -> Result<(), Box<dyn Error>> {
        let out_file = std::fs::File::create(fname)?;
        let _ = serde_json::to_writer_pretty(&out_file, self)?;
        Ok(())
    }

    fn from_json(fname: &PathBuf) -> Result<JobContext, Box<dyn Error>> {
        let in_file = std::fs::File::open(fname)?;
        let job_context: JobContext = serde_json::from_reader(&in_file)?;
        Ok(job_context)
    }

    fn from_jobrow(job_row: JobRow) -> Result<JobContext, Box<dyn Error>> {
        /////////////////////////////////////////////////////////
        // I need a workaround for when the directory for the job is deleted
        /////////////////////////////////////////////////////////
        let mut job_context = JobContext::check_job(&job_row.id)?;
        job_context.job_row = Some(job_row);
        Ok(job_context)
    }

    /// Generate a unique ID with `size` characters. For readability,
    /// the characters used are from the sets [0-9], [A-Z], [a-z].
    pub fn make_id(size: usize) -> String {
        const BASE62: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

        let mut id = String::with_capacity(size);
        let mut rng = rand::thread_rng();
        for _ in 0..size {
            id.push(BASE62[rng.gen::<usize>() % 62] as char);
        }
        id
    }

    async fn set_up(
            db: &Db,
            cookies: &CookieJar<'_>,
            sub: &Submit,
    ) -> Result<JobContext, Box<dyn Error>> {

        let id = JobContext::make_id(JOB_ID_LENGTH);

        let maybe_cookie = cookies.get_private("uid");
        let uid_result: Result<i32, NoLoggedInUserError> = if let Some(cookie) = maybe_cookie {
            let uid: i32 = cookie.value().to_string().parse()?;
            sub.insert_into_db(&db, &uid, &id).await.unwrap();
            Ok(uid)
        } else {
            println!("No uid");
            Err(NoLoggedInUserError)
        };
        let uid = uid_result?;

        let job_path = build_job_path(&id);
        let _ = std::fs::create_dir_all(job_path.clone());
 
        let mut path = PathBuf::new();
        path.push(&job_path);

        // we write the fasta and score files using a uid
        //let email: String = sub.account.email.clone();

        let fa_path = path.join(String::from("seqs.fa"));
        let _ = sub.submission.fa_file.upload(&fa_path).await?;
        // write score file to job's data path
        let score_path = path.join(String::from("scores.txt"));
        let _ = sub.submission.score_file.upload(&score_path).await?;
        let status = JobStatus::Queued;
        Ok(JobContext{
            id: Some(id),
            path,
            fa_path,
            score_path,
            status,
            job_row: None,
        })
    }

    /// creates job data directory with uid, uploads fasta and score files
    async fn set_up_job(&self, sub: &Submit) -> Result<(), Box<dyn Error>> {

        let mut cmd = sub.cfg.build_cmd(
            &self.path,
            &self.fa_path,
            &self.score_path,
        )?;
        let child = spawn_job(&mut cmd).await?;

        Ok(())
    }

    //////////////////////////////////////////////////////////////////
    // I think I should change this to return Option<JobContext>
    // That would simplify cases where errors killed jobs prior to a job_status.json file
    // being created, as I could return None, and handle that None appropriately
    // moving up.
    //////////////////////////////////////////////////////////////////
    pub fn check_job(job_id: &str) -> Result<JobContext, Box<dyn Error>> {

        let job_path = build_job_path(job_id);
        let status_fname = job_path.join("job_status.json");
        //println!("Status file name: {:?}", &status_fname);
        let status_file = std::fs::File::open(&status_fname)?;
        let status = serde_json::from_reader(&status_file)?;
        let fa_path = job_path.join("seqs.fa");
        let score_path = job_path.join("scores.txt");

        Ok(JobContext{
            id: Some(job_id.to_string()),
            path: job_path.into(),
            fa_path: fa_path,
            score_path: score_path,
            status: status,
            job_row: None,
        })
    }
}


//impl Job {
//    ///// creates job data directory with uid, uploads fasta and score files
//    //async fn set_up_job(sub: &Submit, context: &JobContext) -> Result<(), Box<dyn Error>> {
//
//    //    let mut cmd = sub.cfg.build_cmd(
//    //        &context.path,
//    //        &context.fa_path,
//    //        &context.score_path,
//    //    )?;
//    //    let child = spawn_job(&mut cmd).await?;
//
//    //    Ok(())//Job { context, child })
//    //}
//
//    //pub fn check_status(&mut self) -> JobStatus {
//    //    let res = self.child.try_wait();
//    //    let status = match res {
//    //        Ok(no_err) => {
//    //            if let Some(exit_status) = no_err {
//    //                JobStatus::FinishedWithMotifs
//    //            } else {
//    //                JobStatus::Running
//    //            }
//    //        },
//    //        Err(error) => JobStatus::FinishedError,
//    //    };
//    //    status
//    //}
//}

fn build_job_path(id: &str) -> PathBuf {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/", "data");
    Path::new(root).join(id.clone())
}

async fn spawn_job(cmd: &mut Command) -> Result<Child, Box<dyn Error>> {

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
// I need to get writing stdout/stderr to a file working
// HAVE CHILD OPEN/WRITE OUT/ERR
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
    //let log_fname = path.as_path().join("shapeme.log");
    //let mut out_log = tokio::fs::File::create(log_fname).await?;

    //let mut cmd = self.build_cmd()?;

    println!("{:?}", cmd);
    let child = cmd
        //.stdout(out_log)
        .spawn()
        .expect("Error spawning child process");
    //let stdout = child.stdout.take().unwrap();
    //tokio::spawn(async move {
    //    out_log.write(stdout).await.unwrap();
    //});

    Ok(child)
}

/// Creates and spawns a job and inserts into Runs
pub async fn run_job(
        sub: &Submit,
        context: &JobContext,
        //state: &State<Arc<Runs>>,
) -> Result<(), Box<dyn Error>> {
    // set paths and upload files
    let _ = context.set_up_job(sub).await?;

    Ok(())
}

#[derive(Debug)]
pub struct File {
    //file_name: Option<FileName>,
    data: Vec<u8>,
}

impl File {
    pub async fn upload(&self, path: &PathBuf) -> Result<(), Box<dyn Error>> {
        tokio::fs::write(path.clone(), self.data.to_vec()).await?;
        Ok(())
    }
}

#[rocket::async_trait]
impl<'a> FromFormField<'a> for File {
    /////////////////////////////////////////////////////////////////
    // NOTE: write in checks for fasta and score file specification
    /////////////////////////////////////////////////////////////////
    async fn from_data(field: DataField<'a, '_>) -> rocket::form::Result<'a, Self> {
    //async fn from_data(field: DataField) -> rocket::form::Result<Self> {
        let stream = field.data.open(u32::MAX.bytes());
        let bytes = stream.into_bytes().await?;
        let file = File {
            //file_name: field.file_name,
            data: bytes.value,
        };
        Ok(file)
    }
}

////////////////////////////////////////////////////
// I should probably switch all these fields to custom structs that derive FromFormField
////////////////////////////////////////////////////
#[derive(Debug, FromForm)]
#[allow(dead_code)]
pub struct Cfg {
    force: bool,
    skip_inference: bool,
    #[field(validate = range(1..10), default=5)]
    crossval_folds: u8,
    score_file: Option<String>,
    data_dir: Option<String>,
    #[field(validate = range(5..20), default=10)]
    kmer: u8,
    #[field(validate = range(1..5), default=1)]
    max_count: u8,
    continuous: Option<u8>,
    #[field(default=2.0)]
    threshold_sd: f64,
    #[field(validate = range(500..50000), default=500)]
    init_threshold_seed_num: u64,
    #[field(default=100)]
    init_threshold_recs_per_seed: u64,
    #[field(default=2)]
    init_threshold_windows_per_record: u64,
    #[field(default=8)]
    max_batch_no_new_seed: u8,
    #[field(default=4)]
    nprocs: u16,
    #[field(default=3.0)]
    upper_threshold_constraint: f64,
    #[field(default=0.0)]
    lower_threshold_constraint: f64,
    #[field(default=4.0)]
    upper_shape_constraint: f64,
    #[field(default=-4.0)]
    lower_shape_constraint: f64,
    #[field(default=4.0)]
    upper_weights_constraint: f64,
    #[field(default=-4.0)]
    lower_weights_constraint: f64,
    #[field(default=1.0)]
    temperature: f64,
    #[field(default=0.01)]
    t_adj: f64,
    #[field(default=0.25)]
    stepsize: f64,
    #[field(default=20000)]
    opt_niter: u64,
    #[field(default=0.1)]
    alpha: f64,
    #[field(default=500)]
    batch_size: u64,
    find_seq_motifs: bool,
    no_shape_motifs: bool,
    seq_fasta: Option<String>,
    #[field(default=String::from("9"))]
    seq_motif_positive_cats: Option<String>,
    #[field(default=0.05)]
    streme_thresh: f64,
    seq_meme_file: Option<String>,
    shape_rust_file: Option<String>,
    write_all_files: bool,
    exhaustive: bool,
    #[field(default=100000)]
    max_n: u64,
    #[field(default="info")]
    log_level: String,
}

impl Cfg {
    fn build_arg_string(&self) -> Result<String, Box<dyn Error>> {
        let mut args = Vec::new();

        let streme_thresh = format!("{}", self.streme_thresh);
        if self.find_seq_motifs {
            args.push("--find_seq_motifs");
            args.push("--streme_thresh");
            args.push(&streme_thresh);
        }
        let a = if let Some(arg) = &self.seq_motif_positive_cats {
            args.push("--seq_motif_positive_cats");
            format!("{}", arg)
        } else {
            String::from("")
        };
        args.push(&a);
        if self.no_shape_motifs {
            args.push("--no_shape_motifs");
        }

        if let Some(arg) = &self.seq_meme_file {
            args.push("--seq_meme_file");
            args.push(arg);
        }
        if let Some(arg) = &self.shape_rust_file {
            args.push("--shape_rust_file");
            args.push(arg);
        }
        if self.write_all_files {
            args.push("--write_all_files");
        }
        if self.exhaustive {
            args.push("--exhaustive");
        }

        if self.force {
            args.push("--force");
        }
        if self.skip_inference {
            args.push("--skip_inference");
        }
        args.push("--crossval_folds");
        let a = format!("{}", self.crossval_folds);
        args.push(&a);

        args.push("--kmer");
        let a = format!("{}", self.kmer);
        args.push(&a);

        args.push("--max_count");
        let a = format!("{}", self.max_count);
        args.push(&a);

        let a = if let Some(arg) = &self.continuous {
            args.push("--continuous");
            format!("{}", arg)
        } else {
            String::from("")
        };
        args.push(&a);

        args.push("--threshold_sd");
        let a = format!("{}", self.threshold_sd);
        args.push(&a);

        args.push("--init_threshold_seed_num");
        let a = format!("{}", self.init_threshold_seed_num);
        args.push(&a);

        args.push("--init_threshold_recs_per_seed");
        let a = format!("{}", self.init_threshold_recs_per_seed);
        args.push(&a);

        args.push("--init_threshold_windows_per_record");
        let a = format!("{}", self.init_threshold_windows_per_record);
        args.push(&a);

        args.push("--max_batch_no_new_seed");
        let a = format!("{}", self.max_batch_no_new_seed);
        args.push(&a);

        args.push("--nprocs");
        let a = format!("{}", self.nprocs);
        args.push(&a);

        args.push("--threshold_constraints");
        let a = format!("{}", self.lower_threshold_constraint);
        args.push(&a);
        let a = format!("{}", self.upper_threshold_constraint);
        args.push(&a);

        args.push("--shape_constraints");
        let a = format!("{}", self.lower_shape_constraint);
        args.push(&a);
        let a = format!("{}", self.upper_shape_constraint);
        args.push(&a);

        args.push("--weights_constraints");
        let a = format!("{}", self.lower_weights_constraint);
        args.push(&a);
        let a = format!("{}", self.upper_weights_constraint);
        args.push(&a);

        args.push("--temperature");
        let a = format!("{}", self.temperature);
        args.push(&a);

        args.push("--t_adj");
        let a = format!("{}", self.t_adj);
        args.push(&a);

        args.push("--stepsize");
        let a = format!("{}", self.stepsize);
        args.push(&a);

        args.push("--opt_niter");
        let a = format!("{}", self.opt_niter);
        args.push(&a);

        args.push("--alpha");
        let alpha = format!("{}", self.alpha);
        args.push(&alpha);

        args.push("--batch_size");
        let batch_size = format!("{}", self.batch_size);
        args.push(&batch_size);

        args.push("--max_n");
        let max_n = format!("{}", self.max_n);
        args.push(&max_n);

        args.push("--log_level");
        args.push(self.log_level.as_str());

        let arg_string = args.join(" ");
        Ok(arg_string)
    }

    pub fn build_cmd(
            &self,
            job_path: &PathBuf,
            fa_path: &PathBuf,
            score_path: &PathBuf,
    ) -> Result<Command, Box<dyn Error>> {

        //let container =  concat!(
        //    env!("CARGO_MANIFEST_DIR"), "/../../../singularity/current/ShapeME.sif"
        //);
        let pycmd = concat!(env!("CARGO_MANIFEST_DIR"), "/../../python3/ShapeME.py");

        //let mut cmd = Command::new("singularity");
        //cmd.arg("exec");
        //cmd.arg(container);
        //cmd.arg("python");
        //cmd.arg(pycmd);
        
        let mut cmd = Command::new("python");
        cmd.arg(pycmd);

        cmd.arg("--score_file");
        if let Some(arg) = &self.score_file {
            cmd.arg(arg);
        } else {
            cmd.arg(score_path.clone().into_os_string());
        }

        cmd.arg("--data_dir");
        if let Some(arg) = &self.data_dir {
            cmd.arg(arg);
        } else {
            cmd.arg(job_path);
        }

        cmd.arg("--seq_fasta");
        if let Some(arg) = &self.seq_fasta {
            cmd.arg(arg);
        } else {
            cmd.arg(fa_path.clone().into_os_string());
        }

        if self.find_seq_motifs {
            cmd.arg("--find_seq_motifs");
            cmd.arg("--streme_thresh");
            cmd.arg(format!("{}", self.streme_thresh));
        }
        if let Some(arg) = &self.seq_motif_positive_cats {
            cmd.arg("--seq_motif_positive_cats");
            cmd.arg(format!("{}", arg));
        }
        if self.no_shape_motifs {
            cmd.arg("--no_shape_motifs");
        }

        if let Some(arg) = &self.seq_meme_file {
            cmd.arg("--seq_meme_file");
            cmd.arg(arg);
        }
        if let Some(arg) = &self.shape_rust_file {
            cmd.arg("--shape_rust_file");
            cmd.arg(arg);
        }
        if self.write_all_files {
            cmd.arg("--write_all_files");
        }
        if self.exhaustive {
            cmd.arg("--exhaustive");
        }

        if self.force {
            cmd.arg("--force");
        }
        if self.skip_inference {
            cmd.arg("--skip_inference");
        }
        cmd.arg("--crossval_folds");
        cmd.arg(format!("{}", self.crossval_folds));

        cmd.arg("--kmer");
        cmd.arg(format!("{}", self.kmer));

        cmd.arg("--max_count");
        cmd.arg(format!("{}", self.max_count));

        if let Some(arg) = &self.continuous {
            cmd.arg("--continuous");
            cmd.arg(format!("{}", arg));
        }

        cmd.arg("--threshold_sd");
        cmd.arg(format!("{}", self.threshold_sd));

        cmd.arg("--init_threshold_seed_num");
        cmd.arg(format!("{}", self.init_threshold_seed_num));

        cmd.arg("--init_threshold_recs_per_seed");
        cmd.arg(format!("{}", self.init_threshold_recs_per_seed));

        cmd.arg("--init_threshold_windows_per_record");
        cmd.arg(format!("{}", self.init_threshold_windows_per_record));

        cmd.arg("--max_batch_no_new_seed");
        cmd.arg(format!("{}", self.max_batch_no_new_seed));

        cmd.arg("--nprocs");
        cmd.arg(format!("{}", self.nprocs));

        cmd.arg("--threshold_constraints");
        cmd.arg(format!("{}", self.lower_threshold_constraint));
        cmd.arg(format!("{}", self.upper_threshold_constraint));

        cmd.arg("--shape_constraints");
        cmd.arg(format!("{}", self.lower_shape_constraint));
        cmd.arg(format!("{}", self.upper_shape_constraint));

        cmd.arg("--weights_constraints");
        cmd.arg(format!("{}", self.lower_weights_constraint));
        cmd.arg(format!("{}", self.upper_weights_constraint));

        cmd.arg("--temperature");
        cmd.arg(format!("{}", self.temperature));

        cmd.arg("--t_adj");
        cmd.arg(format!("{}", self.t_adj));

        cmd.arg("--stepsize");
        cmd.arg(format!("{}", self.stepsize));

        cmd.arg("--opt_niter");
        cmd.arg(format!("{}", self.opt_niter));

        cmd.arg("--alpha");
        cmd.arg(format!("{}", self.alpha));

        cmd.arg("--batch_size");
        cmd.arg(format!("{}", self.batch_size));

        cmd.arg("--max_n");
        cmd.arg(format!("{}", self.max_n));

        cmd.arg("--log_level");
        cmd.arg(self.log_level.as_str());

        Ok(cmd)
    }
}

#[get("/submit")]
fn submit_form() -> Template {
    Template::render("submit", &Context::default())
}

// NOTE: We use `Contextual` here because we want to collect all submitted form
// fields to re-render forms with submitted values on error. If you have no such
// need, do not use `Contextual`. Use the equivalent of `Form<Submit<'_>>`.
#[post("/submit", data = "<form>")]
async fn submit(
        db: Db,
        form: Form<Contextual<'_, Submit>>,
        cookies: &CookieJar<'_>,
        //runs: &State<Arc<Runs>>,
) -> (Status, Template) {

    let template = match form.value {
        Some(ref submit) => {

            //println!("submit: {:#?}", submit);
            println!("submit.cfg: {:#?}", submit.cfg);

            // start a job, job is placed into managed state
            let job_context = JobContext::set_up(&db, &cookies, submit).await.unwrap();
            let job_id = run_job(
                submit,
                &job_context,
                //&runs,
            ).await.unwrap();
            
            println!("{:?}", form.context);
            //Template::render("success", &form.context)
            Template::render("success", &job_context)
        }
        None => {
            println!("Form: {:?}", &form.context);
            println!("None returned on submit!!");
            Template::render("index", &form.context)
        }
    };

    (form.context.status(), template)
}

/////////////////////////////////////////////////////////////////
// Instead of returning a Template, this function should return html directly, since the reports
// are already written in html files by ShapeME.py
/////////////////////////////////////////////////////////////////
#[get("/jobs/<job_id>")]
async fn get_job(
        job_id: String,
        db: Db,
        mut cookies: &CookieJar<'_>,
//) -> Template {
) -> NamedFile {

    let maybe_cookie = cookies.get_private("uid");
    // if a user is logged in, get their jobs, else redirect to index
    //let template = if let Some(cookie) = maybe_cookie {
    let report = if let Some(cookie) = maybe_cookie {
        let uid = cookie.value().to_string().parse().unwrap();
        let user = User::from_uid(&db, uid).await.unwrap();
        let job_context = user.fetch_job(&db, &job_id).await.unwrap();
        let report = Report::from_file(
            &id,
            &job_context.path,
        ).expect("Unable to generate report");
        //let template = match job_context.status {
        //    JobStatus::FinishedWithMotifs => {
        //        if let Some(id) = job_context.id {
        //            ///////////////////////////////////////////////////////////////
        //            // switch to just reading report.html in main direc, and serving that up here
        //            ///////////////////////////////////////////////////////////////
        //            let report = Report::from_file(
        //                &id,
        //                &job_context.path,
        //            ).expect("Unable to generate report");
        //            //let report = Report::new(
        //            //    &id,
        //            //    &job_context.path,
        //            //).expect("Unable to generate report");
        //            //println!("Rendering job_finished template");
        //            //Template::render("job_finished", &report)
        //            report.page
        //        } else {
        //            //Template::render("job_does_not_exist", &job_context)
        //            Metadata::render("job_does_not_exist", &job_context)
        //        }
        //    },
        //    JobStatus::FinishedNoMotif => Metadata::render("job_no_motif", &job_context),
        //    JobStatus::Running => Metadata::render("job_running", &job_context),
        //    JobStatus::FinishedError => Metadata::render("job_error", &job_context),
        //    JobStatus::Queued => Metadata::render("job_queued", &job_context),
        //    JobStatus::DoesNotExist => Metadata::render("job_does_not_exist", &job_context),
        //};
        //template
    } else {
        println!("No user logged in, redirecting to login page.");
        //Template::render("index", &Context::default())
    };

    //println!("Getting job context");
    //let maybe_cookie = cookies.get_private("uid");
    //let uid_result: Result<i32, NoLoggedInUserError> = if let Some(cookie) = maybe_cookie {
    //    let uid: i32 = cookie.value().to_string().parse()?;
    //    Ok(uid)
    //} else {
    //    println!("No uid");
    //    Err(NoLoggedInUserError)
    //};
    //let uid = uid_result?;

    //let job_context = JobContext::check_job(&job_id).unwrap();
    //template
    report
}

//#[derive(Debug, Serialize)]
#[derive(Debug)]
pub struct Report{
    //id: String,
    //logo_data: String,
    //heatmap_data: String,
    //aupr_curve_data: String,
    //auprs: String,
    pub page: RawHtml<String>,
}

#[derive(Serialize)]
pub struct FoldReport{
    id: String,
    logo_data: Vec<String>,
    heatmap_data: Vec<String>,
    //aupr_curve_data: Vec<String>,
    auprs: String,
}

fn get_img_data(
        job_path: &PathBuf,
        fold_direc: &str,
        img_base: &str,
) -> Result<String, Box<dyn Error>> {
    let img_path = job_path.as_path().join(fold_direc).join(img_base);
    //println!("================");
    //println!("reading image at {:?}", &img_path);
    let img = ImageReader::open(img_path)?.decode()?;

    let mut bytes: Vec<u8> = Vec::new();
    //println!("encoding image as bytes array");
    img.write_to(
        &mut Cursor::new(&mut bytes),
        image::ImageOutputFormat::Png,
    )?;
    //println!("fetching bytes array");
    let data = general_purpose::STANDARD.encode(&bytes);
    Ok(data)
}

fn get_fold_direcs(job_path: &PathBuf) -> Vec<String> {
    let search = job_path
        .as_path()
        .join("*fold_*_output");
    let search_str = search.as_path().to_str().unwrap();
    let fold_dirs = glob(
        search_str
    ).expect("No directories matching *fold_*_output found.");
    let fold_dirs: Vec<String> = fold_dirs.map(
        |a| {
            let res = a.unwrap();
            res.as_path().to_str().unwrap().to_string()
        }
    ).collect();
    fold_dirs
}

impl Report {

    pub fn from_file(id: &str, job_path: &PathBuf) -> Result<Report, Box<dyn Error>> {
        let file_path = job_path.as_path().join("report.html");
        let page_text = std::fs::read_to_string(file_path).unwrap();
        let page = RawHtml(page_text);
        Ok(Report{page})
    }

    //pub fn new(id: &str, job_path: &PathBuf) -> Result<Report, Box<dyn Error>> {

    //    //let job_os_string = job_path.clone().into_os_string();
    //    //let job_direc_string = job_os_string.into_string().unwrap();
    //    //println!("reading logo image");
    //    let logo_data: String = get_img_data(
    //        job_path,
    //        ".",
    //        "final_motifs.png",
    //    ).expect("Unable to open final_motifs.png");

    //    //println!("reading heatmap image");
    //    let heatmap_data: String = get_img_data(
    //        job_path,
    //        ".",
    //        "final_heatmap.png",
    //    ).expect("Unable to open final_heatmap.png");

    //    //println!("reading aupr curve image");
    //    //let aupr_curve_data: String = get_img_data(
    //    //    job_path,
    //    //    ".",
    //    //    "precision_recall_curve.png",
    //    //).expect("Unable to open precision_recall_curve.png");

    //    Ok(Report{
    //        id: String::from(id),
    //        logo_data,
    //        heatmap_data,
    //        //aupr_curve_data,
    //        auprs: String::new(),
    //    })
    //}

}

impl FoldReport {
    pub fn new(id: &str, job_path: &PathBuf) -> Result<FoldReport, Box<dyn Error>> {

        let fold_direcs = get_fold_direcs(&job_path);

        let logo_data: Vec<String> = fold_direcs.iter().map(|fold_direc| {
            get_img_data(
                job_path,
                fold_direc,
                "final_motifs.png",
            ).expect("Unable to open final_motifs.png")
        }).collect();
        let heatmap_data: Vec<String> = fold_direcs.iter().map(|fold_direc| {
            get_img_data(
                job_path,
                fold_direc,
                "final_heatmap.png",
            ).expect("Unable to open final_heatmap.png")
        }).collect();
        //let aupr_curve_data: Vec<String> = fold_direcs.iter().map(|fold_direc| {
        //    get_img_data(
        //        job_path,
        //        fold_direc,
        //        "precision_recall_curve.png",
        //    ).expect("Unable to open precision_recall_curve.png")
        //}).collect();

        Ok(FoldReport{
            id: String::from(id),
            logo_data,
            heatmap_data,
            //aupr_curve_data,
            auprs: String::new(),
        })
    }

}

pub fn stage() -> AdHoc {
    AdHoc::on_ignite("Database stage", |rocket| async {
        rocket.attach(Db::fairing())
            .mount("/", routes![
                login_form, login, create_account_form, create_account,
                submit_form, submit, get_job, dashboard, logout
            ])
    })
}

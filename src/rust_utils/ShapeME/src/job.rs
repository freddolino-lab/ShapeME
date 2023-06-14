use rocket::tokio::process::{Command, Child};
//use rocket::tokio::io::AsyncWriteExt;
use rocket::form::{DataField, FromFormField};
use rocket::FromForm;
use rocket::data::ToByteUnit;
use rocket::tokio;
use rocket::serde::{Serialize, Deserialize};

//use std::sync::Arc;
use std::error::Error;
use std::path::{Path, PathBuf};
//use std::fmt;

use serde_json;
use rand::{self, Rng};
//use dashmap::DashMap;

const JOB_ID_LENGTH: usize = 10;

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

#[derive(Debug, Serialize, Deserialize)]
pub struct JobContext {
    pub id: String,
    //email: String,
    pub path: PathBuf,
    fa_path: PathBuf,
    score_path: PathBuf,
    pub status: JobStatus,
}

#[derive(Debug)]
pub struct Job {
    pub context: JobContext,
    child: Child,
}

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

    pub async fn set_up(sub: &Submit) -> Result<JobContext, Box<dyn Error>> {
        let id = JobContext::make_id(JOB_ID_LENGTH);

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
            id,
            //email,
            path,
            fa_path,
            score_path,
            status,
        })
    }

    pub fn check_job(job_id: &str) -> Result<JobContext, Box<dyn Error>> {

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
// problem here is that I'm not searching the fold directories
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

        let job_path = build_job_path(job_id);
        let status_fname = job_path.join("job_status.json");
        let status_file = std::fs::File::open(&status_fname)?;
        let status = serde_json::from_reader(&status_file)?;
        //let email = "";

        //let (status,email) = if job_path.is_dir() {
        //    ///////////////////////////////////////////////////////////
        //    // needs work
        //    ///////////////////////////////////////////////////////////
        //    let status = JobStatus::FinishedOK;
        //    let email = "no email";
        //    (status,email)
        //} else {
        //    ///////////////////////////////////////////////////////////
        //    // here i should have a 404 catcher
        //    ///////////////////////////////////////////////////////////
        //    (JobStatus::FinishedError, "no email")
        //};

        let fa_path = job_path.join("seqs.fa");
        let score_path = job_path.join("scores.txt");

        Ok(JobContext{
            id: job_id.to_string(),
            path: job_path.into(),
            //email: email.to_string(),
            status: status,
            fa_path: fa_path,
            score_path: score_path,
        })
    }
}


impl Job {
    /// creates job data directory with uid, uploads fasta and score files
    async fn set_up_job(sub: &Submit, context: &JobContext) -> Result<(), Box<dyn Error>> {

        let mut cmd = sub.cfg.build_cmd(
            &context.path,
            &context.fa_path,
            &context.score_path,
        )?;
        let child = spawn_job(&mut cmd).await?;

        Ok(())//Job { context, child })
    }

    pub fn check_status(&mut self) -> JobStatus {
        let res = self.child.try_wait();
        let status = match res {
            Ok(no_err) => {
                if let Some(exit_status) = no_err {
                    JobStatus::FinishedWithMotifs
                } else {
                    JobStatus::Running
                }
            },
            Err(error) => JobStatus::FinishedError,
        };
        status
    }
}

fn build_job_path(id: &str) -> PathBuf {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/", "data");
    Path::new(root).join(id.clone())
}

async fn spawn_job(cmd: &mut Command) -> Result<Child, Box<dyn Error>> {

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
// I need to get writing stdout to a file working
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
    //let context = JobContext::set_up(sub).await?;
    let _ = Job::set_up_job(sub, context).await?;
    //let job_id = context.id.clone();

    //let state_data = state.inner().clone();

    //tokio::spawn(async move {
    //    state_data.dash_map.insert(job_id, job);
    //});

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

//pub struct Runs {
//    pub dash_map: DashMap<String, Job>
//}


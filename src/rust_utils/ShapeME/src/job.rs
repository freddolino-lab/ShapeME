use tokio::process::{Command, Child};

use rocket::form::{DataField, FromFormField};
use rocket::FromForm;
use rocket::data::ToByteUnit;
//use rocket::fs::FileName;
use rocket::{tokio, State};

use std::sync::Arc;
use std::error::Error;
use std::path::{Path, PathBuf};
use rand::{self, Rng};
use dashmap::DashMap;

const JOB_ID_LENGTH: usize = 10;

#[derive(Debug, FromForm)]
struct Password {
    #[field(validate = len(6..))]
    first: String,
    second: String,
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
pub struct Account {
    #[field(validate = len(1..))]
    name: String,
    password: Password,
    #[field(validate = contains('@').or_else(msg!("invalid email address")))]
    pub email: String,
}

#[derive(Debug, FromForm)]
#[allow(dead_code)]
pub struct Submit {
    pub account: Account,
    pub submission: Submission,
    pub cfg: Cfg,
}

#[derive(Debug)]
pub struct Job {
    id: String,
    email: String,
    path: PathBuf,
    fa_path: PathBuf,
    score_path: PathBuf,
    //conf: Cfg,
    child: Child,
}

impl Job {
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

    /// creates job data directory with uid, uploads fasta and score files
    async fn set_up_job(sub: &Submit) -> Result<Job, Box<dyn Error>> {

        let id = Job::make_id(JOB_ID_LENGTH);

        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/", "data");
        let job_path = Path::new(root).join(id.clone());
        let _ = std::fs::create_dir_all(job_path.clone());
 
        let path = job_path;

        // we write the fasta and score files using a uid
        let email: String = sub.account.email.clone();

        let fa_path = path.join(String::from("seqs.fa"));
        let _ = sub.submission.fa_file.upload(&fa_path).await?;
        // write score file to job's data path
        let score_path = path.join(String::from("scores.txt"));
        let _ = sub.submission.score_file.upload(&score_path).await?;

        let mut cmd = sub.cfg.build_cmd(&path, &fa_path, &score_path)?;
        let child = spawn_job(&mut cmd).await?;

        Ok(Job { id, email, path, fa_path, score_path, child })
    }

}

async fn spawn_job(cmd: &mut Command) -> Result<Child, Box<dyn Error>> {

    //let log_fname = self.path.as_path().join("shapeme.log");
    //let out_log = tokio::fs::File::create(log_fname).await?;

    //let mut cmd = self.build_cmd()?;

    println!("{:?}", cmd);
    let child = cmd
        //.stdout(out_log)
        .spawn()
        .expect("Error spawning child process");

    Ok(child)
}

/// Creates and spawns a job and inserts into Runs
pub async fn insert_job(
        sub: &Submit,
        state: &State<Arc<Runs>>,
) -> Result<String, Box<dyn Error>> {
    // set paths and upload files
    let job = Job::set_up_job(sub).await?;
    let job_id = job.id.clone();

    let state_data = state.inner().clone();

    tokio::spawn(async move {
        state_data.dash_map.insert(job.id.clone(), job);
    });

    Ok(job_id)
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
// I should switch all these fields to custom structs that derive FromFormField
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

        let container =  concat!(
            env!("CARGO_MANIFEST_DIR"), "/../../../singularity/current/ShapeME.sif"
        );
        let pycmd = concat!(env!("CARGO_MANIFEST_DIR"), "/../../python3/ShapeME.py");

        let mut cmd = Command::new("singularity");
        cmd.arg("exec");
        cmd.arg(container);
        cmd.arg("python");
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

pub struct Runs {
    pub dash_map: DashMap<String, Job>
}


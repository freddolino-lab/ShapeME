#[macro_use] extern crate rocket;

mod job;
use job::{File, JobId, upload};

use rocket::time::Date;
use rocket::http::{Status, ContentType};
use rocket::form::{Form, Contextual, FromForm, FromFormField, Context};
use rocket::fs::{FileServer, TempFile, relative};

use rocket_dyn_templates::Template;

use serde_json;
use serde::Serialize;
use std::process::{Command, Stdio, ExitStatus, Child};
use std::io::BufWriter;
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use dashmap::DashMap;

const JOB_ID_LENGTH: usize = 10;

#[derive(Debug, FromForm)]
struct Password<'v> {
    #[field(validate = len(6..))]
    #[field(validate = eq(self.second))]
    first: &'v str,
    #[field(validate = eq(self.first))]
    second: &'v str,
}

#[derive(Debug, FromForm)]
#[allow(dead_code)]
struct Submission<'v> {
    #[field(validate = len(1..))]
    name: &'v str,
    //#[field(validate = ext(ContentType::Fasta))]
    fa_file: File<'v>,
    //#[field(validate = ext(ContentType::Scores))]
    score_file: File<'v>,
}

////////////////////////////////////////////////////
// I should switch all these fields to custom structs that derive FromFormField
////////////////////////////////////////////////////
#[derive(Debug, FromForm, Serialize)]
#[allow(dead_code)]
struct Cfg<'v> {
    force: bool,
    skip_inference: bool,
    #[field(validate = range(1..10), default=5)]
    crossval_folds: u8,
    score_file: Option<&'v str>,
    data_dir: Option<&'v str>,
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
    seq_fasta: Option<&'v str>,
    #[field(default="9")]
    seq_motif_positive_cats: Option<&'v str>,
    #[field(default=0.05)]
    streme_thresh: f64,
    seq_meme_file: Option<&'v str>,
    shape_rust_file: Option<&'v str>,
    write_all_files: bool,
    exhaustive: bool,
    #[field(default=100000)]
    max_n: u64,
    #[field(default="info")]
    log_level: &'v str,
}

impl<'a> Cfg<'a> {
    fn build_cmd(&self, job_id: &JobId, fa_path: &PathBuf, score_path: &PathBuf) -> Result<Command, Box<dyn Error>> {

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
        if let Some(arg) = self.score_file {
            cmd.arg(arg);
        } else {
            cmd.arg(score_path.clone().into_os_string());
        }

        cmd.arg("--data_dir");
        if let Some(arg) = self.data_dir {
            cmd.arg(arg);
        } else {
            cmd.arg(job_id.path());
        }

        cmd.arg("--seq_fasta");
        if let Some(arg) = self.seq_fasta {
            cmd.arg(arg);
        } else {
            cmd.arg(fa_path.clone().into_os_string());
        }

        if self.find_seq_motifs {
            cmd.arg("--find_seq_motifs");
            cmd.arg("--streme_thresh");
            cmd.arg(format!("{}", self.streme_thresh));
        }
        if let Some(arg) = self.seq_motif_positive_cats {
            cmd.arg("--seq_motif_positive_cats");
            cmd.arg(format!("{}", arg));
        }
        if self.no_shape_motifs {
            cmd.arg("--no_shape_motifs");
        }

        if let Some(arg) = self.seq_meme_file {
            cmd.arg("--seq_meme_file");
            cmd.arg(arg);
        }
        if let Some(arg) = self.shape_rust_file {
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

        if let Some(arg) = self.continuous {
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
        cmd.arg(self.log_level);

        Ok(cmd)
    }
}

#[derive(Debug, FromForm)]
#[allow(dead_code)]
struct Account<'v> {
    #[field(validate = len(1..))]
    name: &'v str,
    password: Password<'v>,
    #[field(validate = contains('@').or_else(msg!("invalid email address")))]
    email: &'v str,
}

#[derive(Debug, FromForm)]
#[allow(dead_code)]
struct Submit<'v> {
    account: Account<'v>,
    submission: Submission<'v>,
    cfg: Cfg<'v>,
}

#[get("/")]
fn index() -> Template {
    Template::render("index", &Context::default())
}

/////////////////////////////////////////////////////////
// I need to get this output working with rocket ////////
/////////////////////////////////////////////////////////
fn run_job(job: &JobId, conf: &Cfg, fa_path: &PathBuf, score_path: &PathBuf) -> Result<Child, Box<dyn Error>> {

    let conf_fname = job.path().as_path().join("cfg.json");
    // set up writer
    let conf_file = std::fs::File::create(conf_fname).unwrap();
    // open a buffered writer to open the pickle file
    let buf_writer = BufWriter::new(conf_file);
    let _ = serde_json::to_writer_pretty(buf_writer, &conf)?;
 
    let log_fname = job.path().as_path().join("shapeme.log");
    let out_log = std::fs::File::create(log_fname).unwrap();

    let mut cmd = conf.build_cmd(&job, fa_path, score_path)?;
    println!("{:?}", cmd);
    let child = cmd
        .stdout(out_log)
        //.status()?;
        .spawn()
        .expect("Error spawning child process");

    Ok(child)
}

// NOTE: We use `Contextual` here because we want to collect all submitted form
// fields to re-render forms with submitted values on error. If you have no such
// need, do not use `Contextual`. Use the equivalent of `Form<Submit<'_>>`.
// ////////////////////////////////////
// try tokio::spawn for async; need to figure out how to track running jobs
#[post("/", data = "<form>")]
async fn submit<'r>(form: Form<Contextual<'r, Submit<'r>>>) -> (Status, Template, Child) {
    let job_id = JobId::new(JOB_ID_LENGTH);
    let (template,child) = match form.value {
        Some(ref submission) => {
            //println!("submission: {:#?}", submission);
            println!("submission.cfg: {:#?}", submission.cfg);
            //////////////////////////////////////////////////////////
            // here's where I get the job running
            //////////////////////////////////////////////////////////
            // we write the fasta and score files using a uid as their names
            let sub = &submission.submission;
            let cfg = &submission.cfg;
            // write fasta file to job's data path
            let fa_path = upload(&sub.fa_file, String::from("seqs.fa"), &job_id).unwrap();
            // write score file to job's data path
            let score_path = upload(&sub.score_file, String::from("scores.txt"), &job_id).unwrap();
            ////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////
            // I really need to learn how to work with `Status` struct!!!!
            // I need to handle the child process somehow to allow user to query its status
            // in real time, and to send user email when finished
            // Could use done function to dynamically check if job finished
            // if finished without error, serve results template
            // if finished with error, serve log file
            // if not finished, say so
            ////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////
            let child = run_job(&job_id, &cfg, &fa_path, &score_path).unwrap();
            ////////////////////////////////////////////////////
            // I need to handle job_status appropriately at this point
            ////////////////////////////////////////////////////
            
            (Template::render("success", &form.context), Some(child))
        }
        None => {
            println!("None returned on submit!!");
            (Template::render("index", &form.context), None)
        }
    };

    // But what I really need is to just spawn the child
    if let Some(mut child_proc) = child {
        let status = child_proc.wait().expect("Failed to wait on child process");
        if status.success() {
            println!("Job {:?} exited successfully!", job_id);
        } else {
            println!("Job {:?} had non-zero exit status", job_id);
        }
    }
    (form.context.status(), template, child)
}

#[get("/<job_id>")]
fn done(job_id: String) -> Template {
    Template::render("done", &Context::default())
}

#[launch]
fn rocket() -> _ {
    rocket::build()
        .mount("/", routes![index, submit])
        .attach(Template::fairing())
        .mount("/", FileServer::from(relative!("/static")))
}

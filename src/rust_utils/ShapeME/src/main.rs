#[macro_use] extern crate rocket;

mod job;
use job::{File, JobId, upload};

use rocket::time::Date;
use rocket::http::{Status, ContentType};
use rocket::form::{Form, Contextual, FromForm, FromFormField, Context};
use rocket::fs::{FileServer, TempFile, relative};

use rocket_dyn_templates::Template;

const JOB_ID_LENGTH: usize = 10;

#[derive(Debug, FromForm)]
struct Password<'v> {
    #[field(validate = len(6..))]
    #[field(validate = eq(self.second))]
    first: &'v str,
    #[field(validate = eq(self.first))]
    second: &'v str,
}

//#[derive(Debug, FromFormField)]
//enum Rights {
//    Public,
//    Reserved,
//    Exclusive,
//}
//
//#[derive(Debug, FromFormField)]
//enum Category {
//    Biology,
//    Chemistry,
//    Physics,
//    #[field(value = "CS")]
//    ComputerScience,
//}

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

#[derive(Debug, FromForm)]
#[allow(dead_code)]
struct Cfg<'v> {
    force: bool,
    skip_inference: bool,
    #[field(validate = range(1..10))]
    crossval_folds: u8,
    score_file: Option<&'v str>,
    data_dir: Option<&'v str>,
    #[field(validate = range(5..20))]
    kmer: u8,
    #[field(validate = range(1..5))]
    max_count: u8,
    #[field(validate = range(2..20).or_else(msg!("{} must be between 2 and 20", self.continuous)))]
    continuous: u8,
    threshold_sd: f64,
    #[field(validate = range(500..50000).or_else(msg!("{} must be between 500 and 50000", self.init_threshold_seed_num)))]
    init_threshold_seed_num: u64,
    init_threshold_recs_per_seed: u64,
    init_threshold_windows_per_record: u64,
    max_batch_no_new_seed: u8,
    nprocs: u16,
    threshold_constraints: (f64, f64),
    shape_constraints: (f64, f64),
    weights_constraints: (f64, f64),
    temperature: f64,
    t_adj: f64,
    stepsize: f64,
    opt_niter: u64,
    alpha: f64,
    batch_size: u64,
    find_seq_motifs: bool,
    no_shape_motifs: bool,
    seq_fasta: Option<&'v str>,
    seq_motif_positive_cats: Option<&'v str>,
    streme_thresh: f64,
    seq_meme_file: Option<&'v str>,
    shape_rust_file: Option<&'v str>,
    write_all_files: bool,
    exhaustive: bool,
    max_n: u64,
    log_level: &'v str,
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

// NOTE: We use `Contextual` here because we want to collect all submitted form
// fields to re-render forms with submitted values on error. If you have no such
// need, do not use `Contextual`. Use the equivalent of `Form<Submit<'_>>`.
#[post("/", data = "<form>")]
fn submit<'r>(form: Form<Contextual<'r, Submit<'r>>>) -> (Status, Template) {
    let template = match form.value {
        Some(ref submission) => {
            println!("submission: {:#?}", submission);
            //////////////////////////////////////////////////////////
            // here's where I get the job running
            //////////////////////////////////////////////////////////
            // we write the fasta and score files using a uid as their names
            let sub = &submission.submission;
            let job_id = JobId::new(JOB_ID_LENGTH);
            // write fasta file to job's data path
            let fa_path = upload(&sub.fa_file, String::from("seqs.fa"), &job_id);
            // write score file to job's data path
            let score_path = upload(&sub.score_file, String::from("scores.txt"), &job_id);
            Template::render("success", &form.context)
        }
        None => Template::render("index", &form.context),
    };

    (form.context.status(), template)
}

#[launch]
fn rocket() -> _ {
    rocket::build()
        .mount("/", routes![index, submit])
        .attach(Template::fairing())
        .mount("/", FileServer::from(relative!("/static")))
}

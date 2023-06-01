#[macro_use] extern crate rocket;

mod job;
use job::{
    Submit,
    //Runs,
    run_job,
    JobContext,
    Job,
    JobStatus,
};

mod results;
use results::Report;

use rocket::http::Status;
use rocket::form::{Form, Contextual, Context};
use rocket::fs::{FileServer, relative};
use rocket::State;

use rocket_dyn_templates::Template;

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use dashmap::DashMap;

struct SubmitCount {
    count: AtomicUsize
}

#[get("/")]
fn index() -> Template {
    Template::render("index", &Context::default())
}

// NOTE: We use `Contextual` here because we want to collect all submitted form
// fields to re-render forms with submitted values on error. If you have no such
// need, do not use `Contextual`. Use the equivalent of `Form<Submit<'_>>`.
// ////////////////////////////////////
// try tokio::spawn for async; need to figure out how to track running jobs
#[post("/", data = "<form>")]
async fn submit(
        form: Form<Contextual<'_, Submit>>,
        //runs: &State<Arc<Runs>>,
) -> (Status, Template) {

    let template = match form.value {
        Some(ref submission) => {

            //println!("submission: {:#?}", submission);
            println!("submission.cfg: {:#?}", submission.cfg);

            // start a job, job is placed into managed state
            let job_context = JobContext::set_up(submission).await.unwrap();
            let job_id = run_job(
                submission,
                &job_context,
                //&runs,
            ).await.unwrap();
            
            //println!("{:?}", form.context);
            //Template::render("success", &form.context)
            Template::render("success", &job_context)
        }
        None => {
            println!("None returned on submit!!");
            Template::render("index", &form.context)
        }
    };

    (form.context.status(), template)
}

#[get("/jobs/<job_id>")]
async fn get_job(
        job_id: String,
        //runs: &State<Arc<Runs>>,
) -> Template {

    //let data = runs.inner();
    //let job_from_pool = data.dash_map.get_mut(&job_id);

    //println!("{:?}", job_from_pool);
    /////////////////////////////////////////////////////////////////
    // I'll be switching away from the dashmap-based monitoring of current jobs, and toward just
    // checking whether directories exist and if so, does it contain the expected files for a
    // finished job?
    /////////////////////////////////////////////////////////////////
    // If the job is in the current pool it will be Some here.
    // If the job was run in a prior instance, it will be None, so the else block will
    //   take effect
    //let template = 
    //    if let Some(mut job) = job_from_pool {

    //        let report = Report::new(
    //            &job.context.id,
    //            &job.context.path,
    //        ).expect("Unable to generate report");
    //        Template::render("job_finished", &report)

    //    } else {
    let job_context = JobContext::check_directory(&job_id).unwrap();
    let template = match job_context.status {
        JobStatus::FinishedWithMotifs => {
            let report = Report::new(
                &job_context.id,
                &job_context.path,
            ).expect("Unable to generate report");
            Template::render("job_finished", &report)
        },
        JobStatus::FinishedNoMotif => Template::render("job_no_motif", &job_context),
        JobStatus::Running => Template::render("job_running", &job_context),
        JobStatus::FinishedError => Template::render("job_error", &job_context),
        JobStatus::Queued => Template::render("job_queued", &job_context),
        JobStatus::DoesNotExist => Template::render("job_does_not_exist", &job_context),
    };
    template
}

//#[get("/jobs")]
//fn see_jobs(runs: &State<Arc<Runs>>) -> Template {
//    let data = runs.inner();
//    Template::render("jobs", &data)
//}

#[launch]
fn rocket() -> _ {
    rocket::build()
        //.manage(Arc::new(Runs {
        //    dash_map: DashMap::new(),
        //}))
        .manage(SubmitCount {count: AtomicUsize::new(0)})
        .mount("/", routes![index, submit, get_job])
        .attach(Template::fairing())
        .mount("/", FileServer::from(relative!("/static")))
}

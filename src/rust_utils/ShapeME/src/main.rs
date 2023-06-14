#[macro_use] extern crate rocket;

mod job;
use job::{
    Submit,
    //Runs,
    run_job,
    JobContext,
    //Job,
    JobStatus,
};

mod results;
use results::Report;

mod db;

use rocket::http::Status;
use rocket::form::{Form, Contextual, Context};
use rocket::fs::{FileServer, relative};
use rocket::State;

use rocket_dyn_templates::Template;

//use std::sync::Arc;
//use std::sync::atomic::AtomicUsize;
//use dashmap::DashMap;

#[get("/submit")]
fn submit_form() -> Template {
    Template::render("submit", &Context::default())
}

// NOTE: We use `Contextual` here because we want to collect all submitted form
// fields to re-render forms with submitted values on error. If you have no such
// need, do not use `Contextual`. Use the equivalent of `Form<Submit<'_>>`.
#[post("/submit", data = "<form>")]
async fn submit(
        form: Form<Contextual<'_, Submit>>,
        //runs: &State<Arc<Runs>>,
) -> (Status, Template) {

    let template = match form.value {
        Some(ref submit) => {

            //println!("submit: {:#?}", submit);
            println!("submit.cfg: {:#?}", submit.cfg);

            // start a job, job is placed into managed state
            let job_context = JobContext::set_up(submit).await.unwrap();
            let job_id = run_job(
                submit,
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
) -> Template {

    let job_context = JobContext::check_job(&job_id).unwrap();
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
        //.manage(SubmitCount {count: AtomicUsize::new(0)})

        .mount("/", routes![submit_form, submit, get_job])
        .attach(Template::fairing())
        .attach(db::stage())
        .mount("/", FileServer::from(relative!("/static")))
        //.attach(job::stage())
}

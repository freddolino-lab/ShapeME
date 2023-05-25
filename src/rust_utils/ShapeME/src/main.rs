#[macro_use] extern crate rocket;

mod job;
use job::{Submit, Runs, insert_job, JobContext};

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
        runs: &State<Arc<Runs>>,
) -> (Status, Template) {

    let template = match form.value {
        Some(ref submission) => {

            //println!("submission: {:#?}", submission);
            println!("submission.cfg: {:#?}", submission.cfg);

            // start a job, job is placed into managed state
            let job_id = insert_job(submission, &runs).await.unwrap();
            
            println!("{:?}", form.context);
            Template::render("success", &form.context)
        }
        None => {
            println!("None returned on submit!!");
            Template::render("index", &form.context)
        }
    };

    (form.context.status(), template)
}

#[get("/jobs/<job_id>")]
fn get_job(job_id: String, runs: &State<Arc<Runs>>) -> Template {
    let data = runs.inner();
    let job_from_pool = data.dash_map.get_mut(&job_id);
    if let Some(mut job) = job_from_pool {
        let msg = job.check_status();
        println!("{}", msg);
        let job_context = JobContext::from(job);
        ///////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////
        // build context out for the job //////////////////////
        ///////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////
        Template::render("finished", &job_context)
    } else {
        Template::render("job_not_found", &Context::default())
    }
}

//#[get("/jobs")]
//fn see_jobs(runs: &State<Arc<Runs>>) -> Template {
//    let data = runs.inner();
//    Template::render("jobs", &data)
//}

#[launch]
fn rocket() -> _ {
    rocket::build()
        .manage(Arc::new(Runs {
            dash_map: DashMap::new(),
        }))
        .manage(SubmitCount {count: AtomicUsize::new(0)})
        .mount("/", routes![index, submit, get_job])
        .attach(Template::fairing())
        .mount("/", FileServer::from(relative!("/static")))
}

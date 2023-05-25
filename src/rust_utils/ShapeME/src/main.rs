#[macro_use] extern crate rocket;

mod job;
use job::{Submit, Runs, insert_job};

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
            
            Template::render("success", &form.context)
        }
        None => {
            println!("None returned on submit!!");
            Template::render("index", &form.context)
        }
    };

    // But what I really need is to just spawn the child
    //if let Some(mut child_proc) = child {
    //    let status = child_proc.wait().expect("Failed to wait on child process");
    //    if status.success() {
    //        println!("Job {:?} exited successfully!", job_id);
    //    } else {
    //        println!("Job {:?} had non-zero exit status", job_id);
    //    }
    //}
    (form.context.status(), template)
}

#[get("/<job_id>")]
fn done(job_id: String) -> Template {
    Template::render("done", &Context::default())
}

#[launch]
fn rocket() -> _ {
    rocket::build()
        .manage(Arc::new(Runs {
            dash_map: DashMap::new(),
        }))
        .manage(SubmitCount {count: AtomicUsize::new(0)})
        .mount("/", routes![index, submit])
        .attach(Template::fairing())
        .mount("/", FileServer::from(relative!("/static")))
}

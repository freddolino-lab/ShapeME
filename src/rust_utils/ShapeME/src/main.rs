#[macro_use] extern crate rocket;

mod results;
use results::Report;

mod db;
//use db::{
//    Submit,
//    run_job,
//    JobStatus,
//};

use rocket::http::Status;
use rocket::form::{Form, Contextual, Context};
use rocket::fs::{FileServer, relative};
use rocket::State;

use rocket_dyn_templates::Template;

//use std::sync::Arc;
//use std::sync::atomic::AtomicUsize;
//use dashmap::DashMap;

#[launch]
fn rocket() -> _ {
    rocket::build()
        //.manage(Arc::new(Runs {
        //    dash_map: DashMap::new(),
        //}))
        //.manage(SubmitCount {count: AtomicUsize::new(0)})

        //.mount("/", routes![submit_form, submit, get_job])
        .attach(Template::fairing())
        .attach(db::stage())
        .mount("/", FileServer::from(relative!("/static")))
        //.attach(job::stage())
}

use std::path::Path;
use rocket::tokio::fs::File;

#[macro_use] extern crate rocket;

use rocket::data::{Data, ToByteUnit};
use rocket::http::uri::Absolute;

mod paste_id;
use paste_id::PasteId;

// In real app these should be defined dynamically in a conf
const ID_LENGTH: usize = 3;
const HOST: Absolute<'static> = uri!("http://localhost:8000");

#[post("/", data = "<paste>")]
async fn upload(paste: Data<'_>) -> std::io::Result<String> {
    let id = PasteId::new(ID_LENGTH);
    // Data::open() opens a Data struct as stream
    paste.open(128.kibibytes()).into_file(id.file_path()).await?;
    Ok(uri!(HOST, retrieve(id)).to_string())
}

#[get("/")]
fn index() -> &'static str {
    "
    USAGE

      POST /

          accepts raw data in the body of the request and responds with a page containing the
          body's content

      GET /<id>
          
          retrieves the content for the paste with id `<id>`
    "
}

#[get("/<id>")]
async fn retrieve(id: PasteId<'_>) -> Option<File> {
    File::open(id.file_path()).await.ok()
}

#[launch]
fn rocket() -> _ {
    rocket::build()
        .mount("/", routes![index, retrieve, upload])
}

#[macro_use] extern crate rocket;

use rocket::time::Date;
use rocket::http::{Status, ContentType};
use rocket::form::{Form, Contextual, FromForm, FromFormField, Context};
use rocket::fs::{FileServer, TempFile, relative};

use rocket_dyn_templates::Template;

#[derive(Debug, FromForm)]
struct Password<'v> {
    #[field(validate = len(6..))]
    #[field(validate = eq(self.second))]
    first: &'v str,
    #[field(validate = eq(self.first))]
    second: &'v str,
}

#[derive(Debug, FromFormField)]
enum Rights {
    Public,
    Reserved,
    Exclusive,
}

#[derive(Debug, FromFormField)]
enum Category {
    Biology,
    Chemistry,
    Physics,
    #[field(value = "CS")]
    ComputerScience,
}

#[derive(Debug, FromForm)]
#[allow(dead_code)]
struct Submission<'v> {
    #[field(validate = len(1..))]
    title: &'v str,
    date: Date,
    #[field(validate = len(1..=250))]
    r#abstract: &'v str,
    #[field(validate = ext(ContentType::PDF))]
    file: TempFile<'v>,
    #[field(validate = len(1..))]
    category: Vec<Category>,
    rights: Rights,
    ready: bool,
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

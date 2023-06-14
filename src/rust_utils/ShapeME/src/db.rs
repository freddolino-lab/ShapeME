//use std::error::Error;
//use std::io::ErrorKind;
use std::fmt;

use rusqlite::Error;
//use rusqlite::Error::QueryReturnedNoRows;

//use rocket::{Rocket, Build};
use rocket::fairing::AdHoc;
use rocket::serde::{Serialize, Deserialize, json::Json};
use rocket::response::{Debug};//, status::Created};
use rocket::fs::relative;
use rocket::form::{Form, Contextual, Context};
use rocket::FromForm;
use rocket_dyn_templates::Template;
use rocket::http::Status;

//use rocket_sync_db_pools::{database, diesel};
//use self::diesel::prelude::*;

use rocket_sync_db_pools::{database, rusqlite};
use self::rusqlite::params;

#[database("shapeme")]
struct Db(rusqlite::Connection);

//#[database("diesel")]
//struct Db(diesel::SqliteConnection);

type Result<T, E = Debug<rusqlite::Error>> = std::result::Result<T, E>;

#[derive(Debug, FromForm)]
#[allow(dead_code)]
pub struct CreateAccount {
    #[field(validate = len(1..))]
    first: String,
    #[field(validate = len(1..))]
    last: String,
    password: CreatePassword,
    #[field(validate = contains('@').or_else(msg!("invalid email address")))]
    pub email: String,
}

#[derive(Debug, FromForm)]
struct CreatePassword {
    #[field(validate = len(6..))]
    first: String,
    second: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, FromForm)]
#[serde(crate = "rocket::serde")]
pub struct User {
    #[serde(skip_deserializing, skip_serializing_if = "Option::is_none")]
    uid: Option<i32>,
    #[serde(skip_deserializing, skip_serializing_if = "Option::is_none")]
    first: Option<String>,
    #[serde(skip_deserializing, skip_serializing_if = "Option::is_none")]
    last: Option<String>,
    pub email: String,
    password: String,
}

//#[rocket::async_trait]
//impl FromFormField for User {
//    //async fn from_data(field: DataField<'a, '_>) -> rocket::form::Result<'a, Self> {
//    async fn from_data(field: DataField) -> rocket::form::Result<Self> {
//        let stream = field.data.open(u32::MAX.bytes());
//        let bytes = stream.into_bytes().await?;
//        let file = File {
//            //file_name: field.file_name,
//            data: bytes.value,
//        };
//        Ok(file)
//    }
//}


//table! {
//    users (uid) {
//        uid -> Nullable<Integer>,
//        first -> Text,
//        last -> Text,
//        email -> Text,
//        password -> Text,
//    }
//}

#[derive(Debug, Clone)]
struct NoUserError;

impl fmt::Display for NoUserError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "No such user found")
    }
}

#[derive(Debug, Clone)]
struct AuthError;

impl fmt::Display for AuthError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "User and password do not match")
    }
}

async fn check_user(db: &Db, email: String) -> Result<User> {
    let user = db.run(move |conn| {
        conn.query_row("SELECT uid, first, last, email, password FROM users WHERE email = ?1", params![email],
            |r| Ok(User {
                uid: Some(r.get(0)?),
                first: Some(r.get(1)?),
                last: Some(r.get(2)?),
                email: r.get(3)?,
                password: r.get(4)?,
            }))
    }).await?;

    Ok(user)
}

fn check_auth(db: &Db, user: &User, password: &str) -> bool {
    user.password == password
}

#[post("/", data = "<form>")]
async fn login(
        db: Db,
        form: Form<Contextual<'_, User>>,
) -> (Status, Template) {

    //println!("form:\n{:?}", form);
    let template = match form.value {
        Some(ref user) => {
            // check if user exists, if not, serve the create account page
            //let (user_exists,cred_check) = User::check_user(credentials);
            let email = String::from(&user.email);
            let user_info_result = check_user(&db, email).await;
            
            let template = match user_info_result {
                Ok(user_info) => {
                    let authorized = check_auth(&db, &user_info, &user.password);
                    if authorized {
                        println!("User {:?} authorized", &user.email);
                        Template::render("submit", &Context::default())
                    } else {
                        println!("User {:?} not authorized", &user.email);
                        Template::render("no_auth", &form.context)
                    }
                },
                Err(error) => match error.0 {
                    Error::QueryReturnedNoRows => {
                        println!("No such user:\n{:?}", &form.context);
                        Template::render("no_such_user", &form.context)
                    },
                    other_error => {
                        Template::render("index", &Context::default())
                    }
                },
            };
            template
        }
        None => {
            println!("None returned on submit!!");
            Template::render("index", &form.context)
        }
    };
    (form.context.status(), template)
}

#[get("/create_account")]
fn account_form() -> Template {
    Template::render("create_account", &Context::default())
}

#[post("/create_account"), data = "<form>")]
fn create_account(
        db: Db,
        form: Form<Contextual<'_, User>>,
) -> (Status, Template) {
    (form.context.status(), template)
}

pub fn stage() -> AdHoc {
    AdHoc::on_ignite("Database stage", |rocket| async {
        rocket.attach(Db::fairing())
            .mount("/", routes![login, account_form, create_account])
    })
}

use rocket::{Rocket, Build};
use rocket::fairing::AdHoc;
use rocket::serde::{Serialize, Deserialize, json::Json};
use rocket::response::{Debug, status::Created};
use rocket::fs::relative;

use rocket_sync_db_pools::rusqlite;

use self::rusqlite::params;

#[database("rusqlite")]
pub struct Db(rusqlite::Connection);

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(crate = "rocket::serde")]
pub struct User {
    #[serde(skip_deserializing, skip_serializing_if = "Option::is_none")]
    pub id: u64,
    first: String,
    last: String,
    pub email: String,
    password: String,
}

type Result<T, E = Debug<rusqlite::Error>> = std::result::Result<T, E>;

async fn check_user(db: Db) -> Result<Json<Vec<i64>>> {
    let ids = db.run(|conn| {
        conn.prepare("SELECT id FROM posts")?
            .query_map(params![], |row| row.get(0))?
            .collect::<Result<Vec<i64>, _>>()
    }).await?;

    Ok(Json(ids))
}

async fn check_user(db: Db, id: i64) -> Option<Json<Post>> {
    let post = db.run(move |conn| {
        conn.query_row("SELECT id, title, text FROM posts WHERE id = ?1", params![id],
            |r| Ok(Post { id: Some(r.get(0)?), title: r.get(1)?, text: r.get(2)? }))
    }).await.ok()?;

    Some(Json(post))
}

#[delete("/<id>")]
async fn delete(db: Db, id: i64) -> Result<Option<()>> {
    let affected = db.run(move |conn| {
        conn.execute("DELETE FROM posts WHERE id = ?1", params![id])
    }).await?;

    Ok((affected == 1).then(|| ()))
}

#[delete("/")]
async fn destroy(db: Db) -> Result<()> {
    db.run(move |conn| conn.execute("DELETE FROM posts", params![])).await?;

    Ok(())
}

pub fn stage() -> AdHoc {
    AdHoc::on_ignite("Rusqlite Stage", |rocket| async {
        rocket.attach(Db::fairing())
            .attach(AdHoc::on_ignite("Rusqlite Init", init_db))
            .mount("/rusqlite", routes![list, create, read, delete, destroy])
    })
}

use rocket::form::{DataField, FromFormField, Form};
use rocket::http::Status;
use rocket::FromForm;
use rocket::data::ToByteUnit;
use rocket::fs::FileName;
use rocket::response::content::RawHtml;

use std::error::Error;
use std::path::{Path, PathBuf};
use std::borrow::Cow;
use rand::{self, Rng};
//use crate::utils::Md5;

#[derive(Debug)]
pub struct Job<'a> {
    id: JobId<'a>,
}

/// A unique job ID.
#[derive(UriDisplayPath, Debug)]
pub struct JobId<'a>(Cow<'a, str>);

impl JobId<'_> {
    /// Generate a unique ID with `size` characters. For readability,
    /// the characters used are from the sets [0-9], [A-Z], [a-z].
    pub fn new(size: usize) -> JobId<'static> {
        const BASE62: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

        let mut id = String::with_capacity(size);
        let mut rng = rand::thread_rng();
        for _ in 0..size {
            id.push(BASE62[rng.gen::<usize>() % 62] as char);
        }

        JobId(Cow::Owned(id))
    }

    /// Returns the path to the job directory in `data/` corresponding to this ID.
    pub fn path(&self) -> PathBuf {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/", "data");
        let job_path = Path::new(root).join(self.0.as_ref());
        std::fs::create_dir_all(job_path.clone());
        job_path
    }
}

#[derive(Debug)]
pub struct File<'a> {
    file_name: Option<&'a FileName>,
    data: Vec<u8>,
}

impl<'a> File<'a> {
    fn write(&self, path: PathBuf) -> Result<(), Box<dyn Error>> {
        std::fs::write(path.clone(), self.data.to_vec())?;
        Ok(())
    }
}

#[rocket::async_trait]
impl<'a> FromFormField<'a> for File<'a> {
    /////////////////////////////////////////////////////////////////
    // NOTE: write in checks for fasta and score file specification
    /////////////////////////////////////////////////////////////////
    async fn from_data(field: DataField<'a, '_>) -> rocket::form::Result<'a, Self> {
        let stream = field.data.open(u32::MAX.bytes());
        let bytes = stream.into_bytes().await?;
        let file = File {
            file_name: field.file_name,
            data: bytes.value,
        };
        Ok(file)
    }
}

////////////////////////////////////
// can update to async with tokio::fs
////////////////////////////////////
// errors are NOT being handled correctly. If file not able to write, still "success"
////////////////////////////////////
pub fn upload( file: &File<'_>, base: String, job: &JobId ) -> Result<PathBuf, Box<dyn Error>> {

    let path = job.path().as_path().join(base);
    file.write(path.clone())?;

    //Ok(RawHtml(format!("file: {}",
    //    file.file_name.unwrap().as_str().unwrap_or("Frack ")
    //    //&file.data.md5()
    //)))
    Ok(path)
}

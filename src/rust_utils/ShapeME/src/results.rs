use rocket::fs::{FileServer, relative};
use rocket::serde::Serialize;
use rocket::fs::NamedFile;

use std::error::Error;
use std::path::{Path, PathBuf};

#[derive(Serialize)]
pub struct Report{
    logo_file: NamedFile,
    heatmap_file: NamedFile,
}

impl Report {
    pub fn new(job_path: &PathBuf) -> Result<Report, Box<dyn Error>> {
        let logo_file = match std::fs::read(
            job_path.as_path().join("shape_fold_0_output/final_motifs.png")
        ) {
            Ok(image_content) => {

            }
        }
        //let logo_file = NamedFile::open(
        //    job_path.as_path().join("shape_fold_0_output/final_motifs.png")
        //).await?;
        let heatmap_file = match std::fs::read(
            job_path.as_path().join("shape_fold_0_output/final_motifs.png")
        ) {
            Ok(image_content) => {

            }
        }
        let heatmap_file = NamedFile::open(
            job_path.as_path().join("shape_fold_0_output/final_heatmap.png")
        ).await?;
        Ok(Report{
            logo_file,
            heatmap_file,
        })
    }
}

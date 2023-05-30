use rocket::fs::{FileServer, relative};
use rocket::serde::Serialize;
use rocket::fs::NamedFile;

use std::error::Error;
use std::path::{Path, PathBuf};
use std::io::Cursor;
use image::io::Reader as ImageReader;
use glob::glob;
use base64;

#[derive(Serialize)]
pub struct Report{
    id: String,
    logo_data: Vec<String>,
    heatmap_data: Vec<String>,
    aupr_curve_data: Vec<String>,
    auprs: String,
}

fn get_img_data(
        job_path: &PathBuf,
        fold_direc: &str,
        img_base: &str,
) -> Result<String, Box<dyn Error>> {
    let img = ImageReader::open(
        job_path.as_path()
        .join(fold_direc)
        .join(img_base)
    )?.decode()?;

    let mut bytes: Vec<u8> = Vec::new();
    img.write_to(
        &mut Cursor::new(&mut bytes),
        image::ImageOutputFormat::Png,
    )?;
    let data = base64::encode(&bytes);
    Ok(data)
}

fn get_fold_direcs(job_path: &PathBuf) -> Vec<String> {
    let search = job_path
        .as_path()
        .join("*fold_*_output");
    let search_str = search.as_path().to_str().unwrap();
    let fold_dirs = glob(
        search_str
    ).expect("No directories matching *fold_*_output found.");
    let fold_dirs: Vec<String> = fold_dirs.map(
        |a| {
            let res = a.unwrap();
            res.as_path().to_str().unwrap().to_string()
        }
    ).collect();
    fold_dirs
}

//fn get_fold_curves(fold_dirs: Vec<String>) -> Vec<String> {
//    let fold_curves
//}

impl Report {
    pub fn new(id: &str, job_path: &PathBuf) -> Result<Report, Box<dyn Error>> {

        let fold_direcs = get_fold_direcs(&job_path);

        let logo_data: Vec<String> = fold_direcs.iter().map(|fold_direc| {
            get_img_data(
                job_path,
                fold_direc,
                "final_motifs.png",
            ).unwrap()
        }).collect();
        let heatmap_data: Vec<String> = fold_direcs.iter().map(|fold_direc| {
            get_img_data(
                job_path,
                fold_direc,
                "final_heatmap.png",
            ).unwrap()
        }).collect();
        let aupr_curve_data: Vec<String> = fold_direcs.iter().map(|fold_direc| {
            get_img_data(
                job_path,
                fold_direc,
                "precision_recall_curve.png",
            ).unwrap()
        }).collect();

        Ok(Report{
            id: String::from(id),
            logo_data,
            heatmap_data,
            aupr_curve_data,
            auprs: String::new(),
        })
    }
}

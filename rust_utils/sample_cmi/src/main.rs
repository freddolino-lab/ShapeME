use motifer;
use fitting;
use ndarray_npy;
use std::fs;
use std::path;
use std::env;
use std::time;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::io::{BufReader, BufWriter};
use serde_pickle::{de, ser};

fn write_pickle(fpath: &path::Path, data: &Vec<f64>) {
    // set up writer
    let file = fs::File::create(fpath).unwrap();
    // open a buffered writer to open the pickle file
    let mut buf_writer = BufWriter::new(file);
    // write to the writer
    let res = ser::to_writer(
        &mut buf_writer, 
        data, 
        ser::SerOptions::new(),
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let cfg = motifer::parse_config(&args);

    ThreadPoolBuilder::new()
        .num_threads(cfg.cores)
        .build_global()
        .unwrap();

    let rec_db = motifer::RecordsDB::new_from_files(
        &cfg.shape_fname,
        &cfg.yvals_fname,
    );

    // Starting here, I need to loop over chunks of rec_db.
    // I'll have to get the first chunk and calculate initial threshold,
    // then use that threshold for all chunks
    let mut cmi_vals = Vec::<f64>::new();
    let mut threshold = 1000.0;
    //let threshold = &cfg.threshold;

    for (i,batch) in rec_db.batch_iter(cfg.batch_size).enumerate() {

        if i != 0 {
            break;
        }

        println!("Making Seeds from batch {} of RecordsDB", i+1);
        let mut seeds = batch.make_seed_vec(cfg.kmer, cfg.alpha);

        if i == 0 {
            println!("Calculating initial threshold");
            threshold = motifer::set_initial_threshold(
                &seeds,
                &rec_db,
                &cfg.seed_sample_size,
                &cfg.records_per_seed,
                &cfg.windows_per_record,
                &cfg.kmer,
                &cfg.alpha,
                &cfg.thresh_sd_from_mean,
            );
            println!("Initial threshold is {}", &threshold);
        }

        let now = time::Instant::now();
        seeds.compute_mi_values(
            &rec_db,
            &threshold,
            &cfg.max_count,
        );
        let duration = now.elapsed().as_secs_f64() / 60.0;
        println!("MI calculation for batch {} took {} minutes.", i+1, duration);

        let mut cmi_sample = motifer::sample_cmi_vals(
            5000, // sample size
            &seeds,
            &rec_db,
            &cfg.max_count,
        );
        cmi_vals.append(&mut cmi_sample);
    }

    println!(
        "Writing samlped CMI values to {:?}",
        &cfg.out_fname,
    );
    write_pickle(&path::Path::new(&cfg.out_fname), &cmi_vals);

    println!(
        "Starting parameter values: alpha = {:.2}, beta = {:.2}",
        &1.0,
        &1.0,
    ); 
    //println!("{}", &cmi_vals.len());
    let (fitted_params, best_score) = fitting::simulated_annealing(
        &cmi_vals, // data
        vec![1.0,150.0], // starting param vals
        vec![0.0,0.0], // lower bounds on each param
        vec![1000.0,1000.0], // upper bounds on each param
        100.0, // temp
        20.0, // step size
        20000, // niter
        &0.001, // t_adj
        &0.001, // step_adj
        &motifer::beta_logp_objective, // objective fn
        &false, // take acceptance criteria from logit of score?
    );
    //println!("{}", &cmi_vals.len());
    println!(
        "Fitted parameter values: alpha = {:.5}, beta = {:.5}",
        &fitted_params[0],
        &fitted_params[1],
    ); 

    let parent = path::Path::new(&cfg.out_fname).parent().unwrap();

    let param_out_fname = parent.join("param_fits.pkl");
    println!(
        "Writing fitted values of alpha and beta to {:?}",
        &param_out_fname,
    );
    write_pickle(&param_out_fname, &fitted_params);

    // get fitted values of cdf implied by params
    let fitted_vals = motifer::get_fitted_beta_pdf(
        &fitted_params,
        &cmi_vals,
    );

    let fitted_vals_out_fname = parent.join("fitted_pdf.pkl");
    println!(
        "Writing fitted values of the cdf to {:?}",
        &fitted_vals_out_fname,
    );
    write_pickle(&fitted_vals_out_fname, &fitted_vals);

    //let ecdf_out_fname = parent.join("empirical_cdf.pkl");
    //let ecdf = motifer::get_ecdf(&cmi_vals, 1.0);
    ////println!("{}", &ecdf.len());
    //println!(
    //    "Writing empirical cdf values to {:?}",
    //    &ecdf_out_fname,
    //);
    //write_pickle(&ecdf_out_fname, &ecdf);
}


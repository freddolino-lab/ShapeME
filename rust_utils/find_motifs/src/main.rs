use motifer;
use optim;
use std::env;
use std::time;
use std::collections::HashMap;
use rayon::ThreadPoolBuilder;
use partial_application::partial;

//  I ran target/release/find_motifs ../test_data/shapes.npy ../test_data/y_vals.npy ../test_data/config.pkl ../test_data/test_output.pkl
// On Jeremy's laptop, run in series:
//   MI calculation took 36.28 minutes 
// Running on lighthouse to see how that does, run in series:
//   MI calculation took 38.79 minutes
// Python's MI calculation on 1 core on lighthouse:
//   MI calculation took 409.50 minutes

fn main() {
    let args: Vec<String> = env::args().collect();
    let cfg = motifer::parse_config(&args);

    ThreadPoolBuilder::new().num_threads(cfg.cores).build_global().unwrap();

    let rec_db = motifer::RecordsDB::new_from_files(
        cfg.shape_fname,
        cfg.yvals_fname,
    );
    let mut seeds = rec_db.make_seed_vec(cfg.kmer, cfg.alpha);

    let threshold = motifer::set_initial_threshold(
        &seeds,
        &rec_db,
        cfg.seed_sample_size,
        cfg.records_per_seed,
        cfg.windows_per_record,
        &cfg.kmer,
        &cfg.alpha,
        cfg.thresh_sd_from_mean,
    );

    let now = time::Instant::now();
    seeds.compute_mi_values(
        &rec_db,
        &threshold,
        &cfg.max_count,
    );
    let duration = now.elapsed().as_secs_f64() / 60.0;
    println!("MI calculation took {:?} minutes.", duration);

    println!("{} seeds prior to CMI-based filtering.", seeds.len());
    let motifs = motifer::filter_seeds(
        &mut seeds,
        &rec_db,
        &threshold,
        &cfg.max_count,
    );
    println!("{} motifs left after CMI-based filtering.", motifs.len());

    //motifer::pickle_motifs(
    //    &motifs,
    //    &cfg.out_fname,
    //);
    //println!("Vector of motifs written to: {}", &cfg.out_fname);
    let shape_lb = -4.0;
    let shape_ub = 4.0;
    let weights_lb = -4.0;
    let weights_ub = 4.0;
    let thresh_lb = 0.0;
    let thresh_ub = 5.0;

    let test_motif = motifs[0];
    let (params,low,up) = motifer::wrangle_params_for_optim(
        &test_motif,
        &shape_lb,
        &shape_ub,
        &weights_lb,
        &weights_ub,
        &thresh_lb,
        &thresh_ub,
    );

////////////////////////////////////////////////////////////////
// instead of this, try optim_objective(params: Vec<f64>, args: HashMap)
////////////////////////////////////////////////////////////////
    //let objective = partial!(
    //    motifer::optim_objective => _, &cfg.kmer, &rec_db<'a>, &cfg.max_count, &cfg.alpha
    //);
    let mut obj_fn_args = HashMap::new();
    obj_fn_args.insert("kmer", &cfg.kmer);
    obj_fn_args.insert("rec_db", &rec_db);
    obj_fn_args.insert("max_count", &cfg.max_count);
    obj_fn_args.insert("alpha", &cfg.alpha);

    let temp = 1.0;
    let step = 0.25;
    
    let mut particle = optim::Particle::new(
        params,
        low,
        up,
        temp,
        step,
        motifer::optim_objective,
        obj_fn_args,
    );

    let n_iter = 100;
    let t_adjust = 0.10;
    let optimized_result = optim::simulated_annealing(
        &mut particle,
        n_iter,
        &t_adjust,
    );
    println!("{:?}", params);
    println!("{:?}", optimized_result);
}


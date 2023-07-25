use motifer;
use optim;
use ndarray_npy;
use std::process;
use std::path;
use std::env;
use std::time;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use rayon::current_num_threads;

fn main() {
    let args: Vec<String> = env::args().collect();
    let cfg = motifer::parse_config(&args).unwrap();

    ThreadPoolBuilder::new()
        .num_threads(cfg.cores)
        .build_global()
        .unwrap();
    println!("\ninfer_motifs binary using {} cores via rayon", current_num_threads());

    let mut rec_db = motifer::RecordsDB::new_from_files(
        &cfg.shape_fname,
        &cfg.yvals_fname,
    );

    // randomize order of records prior to the next steps.
    // This randomization will be undone later after motif optimization.
    rec_db.permute_records();

    // Starting here, I need to loop over chunks of rec_db.
    // I'll have to get the first chunk and calculate initial threshold,
    // then use that threshold for all chunks
    let mut motifs = motifer::Motifs::empty();
    let mut threshold = 1000.0;
    let mut counter = 0;
    let mut prior_len = 0;
    println!(
        "Max number of batches allowed prior to terminating initial seed evaluation: {}\n",
        cfg.max_batch_no_new,
    );

    for (i,batch) in rec_db.batch_iter(cfg.batch_size).enumerate() {
        println!("Making Seeds from batch {} of RecordsDB.", i+1);
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

        println!("Doing MI calculation for batch {}.", i+1);
        let now = time::Instant::now();
        seeds.compute_mi_values(
            &rec_db,
            &threshold,
            &cfg.max_count,
        );
        let duration = now.elapsed().as_secs_f64() / 60.0;
        println!("MI calculation for batch {} took {} minutes.", i+1, duration);

        //seeds.pickle_seeds(&String::from("/home/x-schroeder/pre_filter_seeds.pkl"));
        println!("{} seeds in batch {} prior to CMI-based filtering.", seeds.len(), i+1);
        let these_motifs = motifer::filter_seeds(
            &mut seeds,
            &rec_db,
            &threshold,
            &cfg.max_count,
        );
        println!(
            "{} seeds in batch {} after CMI-based filtering.",
            these_motifs.len(),
            i+1,
        );
        motifs.append(these_motifs);

        // don't waste time doing another cmi filter if we have the max batch num set
        // super high.
        if cfg.max_batch_no_new < 1000000000 {
            motifs = motifs.filter_motifs(
                &rec_db,
                &cfg.max_count,
            );

            // if we added motifs, re-set number of batches without addition counter
            let motifs_len = motifs.len();
            if motifs_len > prior_len {
                prior_len = motifs_len;
                counter = 0;
                println!("At least one motif from batch {} was added to list. Current number of motifs is {}.", i+1, prior_len);
            } else {
                counter += 1;
                if counter == 1 {
                    println!("No new motifs added to list for {} batch.", counter);
                } else {
                    println!("No new motifs added to list for {} batches.", counter);
                }
            }
            if counter == cfg.max_batch_no_new {
                println!("\nLimit on the number of batches of seeds evaulated with no new motif additions reached. Breaking loop of initial seed evaluation.\n");
                break
            }
        }
        println!();
    }

    if motifs.len() == 0 {
        println!("No shape motifs found by infer_motifs binary.");
        process::exit(0x0100);
    }
    println!("{} seeds collected during initial batched evaluation.", motifs.len());
    let mut motifs = motifs.filter_motifs(
        &rec_db,
        &cfg.max_count,
    );
    println!("{} motifs left after pooled CMI-based filtering.", motifs.len());

    // Optimization options
    // set temp to difference between p_temp and 0.5 on logit scale
    let temp = ((0.5+cfg.temperature)/(0.5-cfg.temperature)).ln().abs();
    // if doing an ensemble optimization technique,
    // initial particles are dropped onto the landscape at
    // init_position + Normal(0.0, init_jitter)
    //let init_jitter = &cfg.stepsize * 1.0;
    //let inertia = 0.8;
    //let local_weight = 0.2;
    //let global_weight = 0.8;

    //let n_particles = 100; // ignored if doing simulated annealing
    //let n_iter_exchange = 2;

    motifs.motifs.par_iter_mut()
        .enumerate()
        // can uncomment this line when debugging to just optimize first two motifs
        //.filter(|(i,motif)| i < &2)
        .for_each(|(i,motif)|
    {

        // get the starting MI just to see how the optimization went
        let start_mi = motif.mi;

        let now = time::Instant::now();
        let (params,low,up) = motifer::wrangle_params_for_optim(
            &motif,
            &cfg.shape_lower_bound,
            &cfg.shape_upper_bound,
            &cfg.weight_lower_bound,
            &cfg.weight_upper_bound,
            &cfg.thresh_lower_bound,
            &cfg.thresh_upper_bound,
        );

        //println!("Using replica exchange as the optimization method.");
        //let (optimized_result,optimized_score) = optim::replica_exchange(
        //    params,
        //    low,
        //    up,
        //    n_particles,
        //    n_iter_exchange,
        //    temp,
        //    step,
        //    n_iter,
        //    &t_adjust,
        //    &motifer::optim_objective,
        //    &rec_db,
        //    &cfg.kmer,
        //    &cfg.max_count,
        //    &cfg.alpha,
        //    &true,
        //);

        //println!("Optimizing motif {} using particle swarm.", i);
        //let (optimized_result,optimized_score) = optim::particle_swarm(
        //    params,
        //    low,
        //    up,
        //    n_particles,
        //    inertia,
        //    local_weight,
        //    global_weight,
        //    init_jitter,
        //    n_iter,
        //    &motifer::optim_objective,
        //    &rec_db,
        //    &cfg.kmer,
        //    &cfg.max_count,
        //    &cfg.alpha,
        //    &true,
        //);

        println!("Optimizing motif {} using simulated annealing.", i);
        let (optimized_result,_optimized_score) = optim::simulated_annealing(
            params,
            low,
            up,
            temp,
            cfg.stepsize,
            cfg.n_opt_iter,
            &cfg.t_adjust,
            &motifer::optim_objective,
            &rec_db,
            &cfg.kmer,
            &cfg.max_count,
            &cfg.alpha,
            // true here indicates that we'll use logit-transformed AMI
            // for our acceptance test.
            // The reason for this is that we don't want to make a jump
            // from 0.99 to 0.79 as easy to make as a jump from 0.5 to 0.3.
            &true, 
        );

        let optimized_motif = motifer::opt_vec_to_motif(
            &optimized_result,
            &rec_db,
            &cfg.alpha,
            &cfg.max_count,
            &cfg.kmer,
        );
        let duration = now.elapsed().as_secs_f64() / 60.0;

        println!("Optimizing motif {} took {:?} minutes.\nIt started with an adjusted mutual information of {}, and ended with a value of {}.", i, duration, &start_mi, &optimized_motif.mi);

        *motif = optimized_motif;
    });

    println!("\nFiltering optimized motifs based on conditional mutual information.");
    let mut motifs = motifs.filter_motifs(
        &rec_db,
        &cfg.max_count,
    );
    println!("{} motifs left after CMI-based filtering.", motifs.len());

    // undo record permutation now, prior to final udate of hits/mi/etc.
    rec_db.undo_record_permutation();

    motifs.post_optim_update(&rec_db, &cfg.max_count);
    motifs.json_motifs(&cfg.out_fname);
    let corr = motifs.get_motif_correlations();

    let parent = path::Path::new(&cfg.out_fname).parent().unwrap();
    let corr_out_fname = parent.join("motif_correlations.npy");
    ndarray_npy::write_npy(&corr_out_fname, &corr).unwrap();

    println!("Vector of filtered, optimized motifs written to: {}", &cfg.out_fname);
    println!("Numpy array of motif correlations written to: {:?}", &corr_out_fname);
}


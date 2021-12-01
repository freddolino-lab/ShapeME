use std::error::Error;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::cmp;
use std::iter;
use std::fs;
use std::io::{BufReader, BufWriter};
// allow random iteration over RecordsDB
use rand::thread_rng;
use rand::seq::SliceRandom;
// ndarray stuff
use ndarray::prelude::*;
use ndarray::Array;
// ndarray_stats exposes ArrayBase to useful methods for descriptive stats like min.
use ndarray_stats::QuantileExt;
use itertools::Itertools;
// ln_gamma is used a lot in expected mutual information to compute log-factorials
use statrs::function::gamma::ln_gamma;
use ndarray_npy; // check provenance, see if we can find what may be a more stable implementation. If ndarray_npy stops existing, we don't want our code to break.
use serde_pickle::{de, ser};
use serde::{Serialize, Deserialize};
// parallelization utilities provided by rayon crate
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
// check the source code for OrderedFloat and consider implementing it in lib.rs instead of using the ordered_float crate. If ordered_float stops existing someday, we don't want our code to break.
use ordered_float::OrderedFloat;
use info_theory;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    fn set_up_sequence(val: f64, size: usize) -> Sequence {
        let ep_param = Param::new(ParamType::EP, Array::from_vec(vec![val; size])).unwrap();
        let prot_param = Param::new(ParamType::ProT, Array::from_vec(vec![val; size])).unwrap();
        let helt_param = Param::new(ParamType::HelT, Array::from_vec(vec![val; size])).unwrap();
        let roll_param = Param::new(ParamType::Roll, Array::from_vec(vec![val; size])).unwrap();
        let mgw_param = Param::new(ParamType::MGW, Array::from_vec(vec![val; size])).unwrap();
        let mut seq_vec = Vec::new();
        seq_vec.push(ep_param);
        seq_vec.push(prot_param);
        seq_vec.push(helt_param);
        seq_vec.push(roll_param);
        seq_vec.push(mgw_param);

        // make the Motif struct
        let this_sequence = Sequence::new(seq_vec);
        
        this_sequence.unwrap()
    }

    fn set_up_stranded_sequence(val: f64, size: usize) -> StrandedSequence {
        let fwd_seq = set_up_sequence(val, size);
        let rev_seq = set_up_sequence(val, size);

        let mut arr = ndarray::Array3::zeros(
            (fwd_seq.params.dim().0, fwd_seq.params.dim().1,0)
        );
        arr.append(Axis(2), fwd_seq.params.insert_axis(Axis(2)).view()).unwrap();
        arr.append(Axis(2), rev_seq.params.insert_axis(Axis(2)).view()).unwrap();
        
        let this_sequence = StrandedSequence::new(arr);
        this_sequence
    }

    #[test]
    fn test_count() {
        let vec = array![1,2,3,2,3,4,3,4,4,4];
        let vecview = vec.view();
        let p_i = info_theory::get_probs(vecview);
        println!("{:?}", p_i);
    }

    #[test]
    fn test_categorize_hits() {
        // answer is that the dot products should be after sorting each row of b
        let answer = arr1(&[0, 3, 6, 3, 4, 7, 6, 7, 8]);
        let max_count = 2;
        // example hits array        // fwd dot prod | rev dot prod
        let mut b = arr2(&[[0, 0],   // 0            | 0
                           [1, 0],   // 1            | 3
                           [2, 0],   // 2            | 6
                           [0, 1],   // 3            | 1
                           [1, 1],   // 4            | 4
                           [2, 1],   // 5            | 7
                           [0, 2],   // 6            | 2
                           [1, 2],   // 7            | 5
                           [2, 2]]); // 8            | 8

        sort_hits(&mut b);

        let categories = info_theory::categorize_hits(&b, &max_count);
        assert_eq!(categories, answer);
    }
 
    #[test]
    fn test_window_over_param() {
        let param = Param
            ::new(ParamType::EP, Array::<f64, _>::linspace(0.0, 20.0, 21))
            .unwrap();
        let mut i = 0.0;
        for window in param.windows(5) {
            assert_eq!(window, Array::<f64, _>::linspace(i, i + 4.0, 5));
            i += 1.0;
        }
    }

    #[test]
    fn test_window_over_seq(){
        let this_seq = set_up_sequence(2.0, 30);
        for window in this_seq.window_iter(0, 10, 3) {
            assert_eq!(ndarray::Array2::from_elem((5, 3), 2.0),
                       window.params)
        }
    }
    
    #[test]
    fn test_pairwise_comparison(){
        let this_seq = set_up_sequence(2.0, 30);
        let mut out = Vec::new();
        for window1 in this_seq.window_iter(0, 31, 3) {
            for window2 in this_seq.window_iter(0, 31, 3) {
                out.push(manhattan_distance(&window1.params, &window2.params));
            }
        }
        assert_eq!(out.len(), (30-2)*(30-2));
    }

    #[test]
    fn test_motif_normalization() {
        let length = 12;
        let this_seq = set_up_sequence(2.0, length);

        let mut this_motif = Motif::new(
            this_seq,
            1.0,
            10,
        );

        // check that it initializes to 1
        assert_eq!(this_motif.weights.weights.sum(), 1.0*length as f64*5.0);
        // check that normed weights intialize to 0
        assert_eq!(this_motif.weights.weights_norm.sum(), 0.0);
        this_motif.weights.normalize();
        // check that when normalized, weights sum to 1
        let sum_normed = this_motif.weights.weights_norm.sum();
        assert!(AbsDiff::default().epsilon(1e-6).eq(&sum_normed, &1.0));
        // check that when normalized. Unnormalized weights are untouched
        assert_eq!(this_motif.weights.weights.sum(), 1.0*length as f64*5.0);
    }

    #[test]
    fn test_constrain_norm() {
        let alpha = 0.1;
        let trans_arr = array![
            [0.15, 0.50, 0.95],
            [0.20, 0.70, 0.60]
        ];
        let total = trans_arr.sum();
        let target_arr = &trans_arr / total;
        let start_arr = trans_arr
            .map(|x| ((x-alpha)/(1.0-alpha) / (1.0-(x-alpha)/(1.0-alpha)) as f64).ln());
        let mut motif_weights = MotifWeights{
            weights: start_arr,
            weights_norm: Array::<f64, Ix2>::zeros(trans_arr.raw_dim())
        };
        motif_weights.constrain_normalize(&alpha);
        let sum_normed = motif_weights.weights_norm.sum();
        assert!(AbsDiff::default().epsilon(1e-6).eq(&sum_normed, &1.0));
        assert!(motif_weights.weights_norm.abs_diff_eq(&target_arr, 1e-6));
    }

    #[test]
    fn test_RecordsDB_seq_iter(){
        let this_seq = set_up_stranded_sequence(2.0, 32);
        let this_seq2 = set_up_stranded_sequence(3.0, 20);
        let seq_vec = vec![this_seq, this_seq2];
        let this_db = RecordsDB::new(seq_vec, array![0,1]);
        for (i,entry) in this_db.iter().enumerate() {
            println!("{:?}", entry);
        }
    }

    #[test]
    fn test_get_seeds() {
        let kmer = 15;
        let this_seq = set_up_stranded_sequence(2.0, 30);
        let that_seq = set_up_stranded_sequence(2.0, 60);
        let rec_db = RecordsDB::new(vec![this_seq, that_seq], array![0,1]);
        let seeds = rec_db.make_seed_vec(kmer, 0.01);
        assert_eq!(seeds.seeds.len(), 62)
    }

    #[test]
    fn test_hit_counting(){
        let kmer = 15;
        let threshold1 = 0.0;
        let threshold2 = 1.0;
        let max_count = 2;
        let alpha = 0.01;
        let length = 30;
        let this_seq = set_up_sequence(2.0, kmer);
        let this_view = this_seq.view();

        let that_seq = set_up_sequence(2.1, length);

        let mut seed_weights = MotifWeights::new(&this_view);
        seed_weights.constrain_normalize(&alpha);
        let wv = seed_weights.weights_norm.view();
        
        // seed now owns the view
        let seed = Seed::new(this_view, kmer);

        let hits = that_seq.count_hits_in_seq(
            &seed.params.params,
            &wv,
            &threshold1,
            &max_count,
        );
        assert_eq!(hits[0], 0);
        assert_eq!(hits[1], 0);

        let hits = that_seq.count_hits_in_seq(
            &seed.params.params,
            &wv,
            &threshold2,
            &max_count,
        );
        assert_eq!(hits[0], 2);
        assert_eq!(hits[1], 0);
    }

    fn setup_RecordsDB(num_seqs: usize, length_seqs: usize) -> RecordsDB{
        let mut seqs = Vec::new();
        let mut vals = Vec::new();
        for i in 0..num_seqs{
            seqs.push(set_up_stranded_sequence(1.0 + i as f64, length_seqs));
            if i % 3 == 0 {
                vals.push(1);
            } else {
                vals.push(0);
            }
        }
        RecordsDB::new(seqs, ndarray::Array1::from_vec(vals))
    }

    #[test]
    fn test_recordb_hit_counting(){
        let db = setup_RecordsDB(30, 30);
        let seeds = db.make_seed_vec(15, 0.01);
        let test_seed = &seeds.seeds[100];
        let hits = db.get_hits(&test_seed.params.params,
                    &seeds.weights.weights_norm.view(),
                    &1.0,
                    &10);
        assert_eq!(hits[[6,0]], 10)
    }

    #[test]
    fn test_recordb_ami(){
        let max_count: i64 = 10;
        let db = setup_RecordsDB(30, 30);
        let seeds = db.make_seed_vec(15, 0.01);
        let test_seed = &seeds.seeds[100];
        let hits = db.get_hits(&test_seed.params.params,
                    &seeds.weights.weights_norm.view(),
                    &1.0,
                    &max_count);
        
        let hit_cats = info_theory::categorize_hits(&hits, &max_count);
        let hv = hit_cats.view();
        let vv = db.values.view();
        let contingency = info_theory::construct_contingency_matrix(hv, vv);
        let ami = info_theory::adjusted_mutual_information(contingency.view());
        let ami_answer = 0.04326620722450172; // calculated using sklear.metrics.adjusted_mutual_info_score
        assert!(AbsDiff::default().epsilon(1e-6).eq(&ami_answer, &ami));
    }

    #[test]
    #[ignore]
    fn test_sum_NaN() {
        // just making sure that a vector of entirely NaN values sums to 0.0.
        let nan = f64::NAN;
        let v = vec![nan, nan, nan];
        let s = v.iter().filter(|a| !a.is_nan()).sum::<f64>();
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_read_files() {
        // read in shapes
        let fname = "../../test_data/shapes.npy";
        let arr: Array4<f64> = ndarray_npy::read_npy(fname).unwrap();
        assert_eq!((2000, 5, 56, 2), arr.dim());
        assert!(AbsDiff::default().epsilon(1e-6).eq(&125522.42816848765, &arr.sum()));

        // make a test array for comparing. I looked in python at the values
        // in my test records database at records.X[0,0:3,0:2,:] (note the shape and
        // length axes are permuted in python relative to rust)
        // and copied them here. So we know this is what the values in our records
        // database at slice s![0, 0..2, 0..3, ..] should be.
        let test_arr = ndarray::array![
            [ // first strand
                // first shape | second shape
                [0.14024065, -0.83476579], // first value
                [-0.95497207, -2.22381607], // second value
                [-0.54092823, -1.30891276] // third value
            ],
            [ // second strand
                // first shape | second shape
                [-0.91056253, -0.10117361], // first value
                [ 0.40469446,  0.60029678], // second value
                [ 0.03372454,  1.85484959] // third value
            ]
        ];
        // check that when I slice as below, I get the values in test_arr
        assert!(
            test_arr
            .abs_diff_eq(
                &arr.slice(s![0,0..2,0..3,..]),
                1e-6,
            )
        );

        // read in y-vals
        let fname = "../../test_data/y_vals.npy";
        let y_vals: Array1<i64> = ndarray_npy::read_npy(fname).unwrap();
        assert_eq!((2000), y_vals.dim());
        assert_eq!(391, y_vals.sum());

        // read in hits for first record, first window, calculated in python
        // These values will be used to test whether our rust hit counting
        // yields the same results as our python hit counting.
        let fname = "../../test_data/hits.npy";
        let hits: Array2<i64> = ndarray_npy::read_npy(fname).unwrap();
        assert_eq!((2000,2), hits.dim());
        assert_eq!(1323, hits.sum());

        // read in some other parameters we'll need
        let fname = "../../test_data/test_args.pkl";
        let mut file = fs::File::open(fname).unwrap();
        // open a buffered reader to open the pickle file
        let mut buf_reader = BufReader::new(file);
        // create a hashmap from the pickle file's contents
        let hash: HashMap<String, f64> = de::from_reader(
            buf_reader,
            de::DeOptions::new()
        ).unwrap();

        // set up test case for having read in the data accurately
        let mut answer = HashMap::new();
        answer.insert(String::from("cores"), 4.0);
        answer.insert(String::from("max_count"), 1.0);
        answer.insert(String::from("alpha"), 0.01);
        answer.insert(String::from("kmer"), 15.0);
        answer.insert(String::from("threshold"), 0.8711171869882366);
        answer.insert(String::from("mi"), 0.0020298099145125295);
        answer.insert(String::from("cmi"), 0.0006904180945122333);

        // check if what we read in is what we expect to see
        assert_eq!(answer, hash);
    }

    #[test]
    fn test_db_from_file() {
        // read in shapes
        let shape_fname = String::from("../../test_data/shapes.npy");
        // read in y-vals
        let y_fname = String::from("../../test_data/y_vals.npy");
        let rec_db = RecordsDB::new_from_files(
            shape_fname,
            y_fname,
        );
        for (i,rec) in rec_db.iter().enumerate() {
            if i > 0 {
                break
            }
            let test_arr = ndarray::array![
                [ // first strand
                    // first shape | second shape
                    [0.14024065, -0.83476579], // first value
                    [-0.95497207, -2.22381607], // second value
                    [-0.54092823, -1.30891276] // third value
                ],
                [ // second strand
                    // first shape | second shape
                    [-0.91056253, -0.10117361], // first value
                    [ 0.40469446,  0.60029678], // second value
                    [ 0.03372454,  1.85484959] // third value
                ]
            ];
 
            println!("test arr: {:?}", test_arr);
            println!("stranded seq arr: {:?}", rec.seq.params.slice(s![0..2,0..3,..]));
            // check that when I slice as below, I get the values in test_arr
            assert!(
                test_arr
                .abs_diff_eq(
                    &rec.seq.params.slice(s![0..2,0..3,..]),
                    1e-6,
                )
            );

        }
    }
    
    #[test]
    fn test_manual_dist() {
        ////////////////////////////////////////////////////////////////////
        // to further test distance calculation and broadcasting accuracy,
        //   I'll create the following:
        //
        //   test_arr1 - a 3d array of shape (2,5,2) with values of 0
        //      for first index of final
        //      axis and of 1 for second index of final axis.
        //   test_arr2 - a 2d array of shape (2,5) with values of 0.5
        //   test_weights - a 2d array of shape (2,5) with values of 1.0
        //   answer_arr - a 1d array with values [5.0, 5.0].
        //
        // running the test arrays through my stranded weighted distance
        //   calculation should yield answer_arr
        ////////////////////////////////////////////////////////////////////

        // a is 2x5 and all 0.0
        let a = Array::zeros((2,5));
        // b is 2x5 and all 1.0
        let b = Array::ones((2,5));
        // test_arr1 is 2x5x2
        let test_arr1 = ndarray::stack![Axis(2), a, b];
        // test_arr2 is 2x5 and all 0.5
        let test_arr2 = Array::from_elem((2, 5), 0.5);
        // weights are 0.5 to keep things simple
        let test_weights = Array::from_elem((2, 5), 0.5);
        // each distance should be 2.5
        let answer_arr = Array::from_elem(2, 2.5);
        
        let test_res = stranded_weighted_manhattan_distance(
            &test_arr1.view(), 
            &test_arr2.view(),
            &test_weights.view(),
        );
        assert!(
            test_res
            .abs_diff_eq(
                &answer_arr,
                1e-6,
            )
        );
    }

    #[test]
    fn test_db_operations() {
        //////////////////////////////////////////////////////////////////////////////
        // We're testing several aspects of working with rec_db structs and seeds here.
        // Although the math look right everywhere I've investigated it, our rust
        // implementation's distance calculations comparing a seed to plus strands
        // match the values we get in our python implementation. The minus strand
        // comparisons do not match, however. I do not understand this, especially since
        // I have done tests with simplified examples and achieved correct results, i.e.,
        // matching minus strand distances between python and rust.
        // I'm a somewhat of a loss for ideas on what else to test here.
        //////////////////////////////////////////////////////////////////////////////

        // read in shapes
        let shape_fname = String::from("../../test_data/shapes.npy");
        // read in y-vals
        let y_fname = String::from("../../test_data/y_vals.npy");
        let rec_db = RecordsDB::new_from_files(
            shape_fname,
            y_fname,
        );

        // read in some other parameters we'll need
        let fname = "../../test_data/test_args.pkl";
        let mut file = fs::File::open(fname).unwrap();
        // open a buffered reader to open the pickle file
        let mut buf_reader = BufReader::new(file);
        // create a hashmap from the pickle file's contents
        let hash: HashMap<String, f64> = de::from_reader(
            buf_reader,
            de::DeOptions::new()
        ).unwrap();
    
        let kmer = *hash.get("kmer").unwrap() as usize;
        let alpha = *hash.get("alpha").unwrap();
        let max_count = *hash.get("max_count").unwrap() as i64;
        let threshold = *hash.get("threshold").unwrap();

        let mut seeds = rec_db.make_seed_vec(kmer, alpha);
        let test_seed = &mut seeds.seeds[1];

        let mut dists: ndarray::Array2<f64> = ndarray::Array2::zeros((42, 2));
        let mut minus_dists = ndarray::Array1::zeros(42);
        //////////////////////////////////////////////////////////////////////////////
        // The second record in the database was compared to the second seed in python.
        // So look at second record in rec_db and second seed, just like I did in python
        // when I made the file "../../test_data/distances.npy"
        //////////////////////////////////////////////////////////////////////////////
        for (i,rec) in rec_db.iter().enumerate() {
            // skip all but i == 1
            if i != 1 {
                continue
            }
            for (j,window) in rec.seq.window_iter(0, rec.seq.params.raw_dim()[1]+1, kmer).enumerate() {
                // calculate just some minus strand distances
                let this_minus_dist = weighted_manhattan_distance(
                        &window.params.slice(s![..,..,1]),
                        &test_seed.params.params,
                        &seeds.weights.weights_norm.view(),
                );
                // calculate stranded distances
                let these_dists = stranded_weighted_manhattan_distance(
                        &window.params,
                        &test_seed.params.params,
                        &seeds.weights.weights_norm.view(),
                );
                // place the respective distances into their appropriate containers
                dists.row_mut(j).assign(
                    &these_dists
                );
                minus_dists[j] = this_minus_dist;
            }
        }

        // read in the distances output by python
        let dist_answer: Array2<f64> = ndarray_npy::read_npy(
            String::from("../../test_data/distances.npy")
        ).unwrap();
        //println!("Dist answer: {:?}", dist_answer);
        //println!("Minus dists: {:?}", minus_dists);
        //println!("Distances: {:?}", dists);

        // assert that all but the final 5 plus strand distances are the same
        // between the python and rust implementations.
        // the reason we stop the comparison where we do is that in python,
        // it hit the max count for each strand at that point, so its distances
        // after that point are all 0.0, whereas the way I looped over the
        // sequences above calculated the distances for all sequences.
        assert!(
            dists
            .abs_diff_eq(
                &dist_answer,
                1e-6,
            )
        );
        // assert that the minus strand distances are equal when
        // calculated for just the minus strand, or when sliced from the
        // stranded distance calc results
        assert!(
            minus_dists
            .abs_diff_eq(
                &dists.slice(s![..,1]),
                1e-6,
            )
        );
        // assert that the minus strand distances calculated in rust
        // are equal to those calculated in python.
        assert!(
            minus_dists
            .abs_diff_eq(
                &dist_answer.slice(s![..,1]),
                1e-6,
            )
        );

        // get the test_seed hits in all elements of rec_db
        let mut hits = rec_db.get_hits(
            &test_seed.params.params,
            &seeds.weights.weights_norm.view(),
            &threshold,
            &max_count,
        );
        // sort the columns of each row of hits

        sort_hits(&mut hits);

        let hit_cats = info_theory::categorize_hits(&hits, &max_count);
        let hv = hit_cats.view();
        let vv = rec_db.values.view();
        let contingency = info_theory::construct_contingency_matrix(hv, vv);
        let ami = info_theory::adjusted_mutual_information(contingency.view());

        test_seed.update_info(
            &rec_db,
            &seeds.weights.weights_norm.view(),
            &threshold,
            &max_count,
        );
        assert!(AbsDiff::default().epsilon(1e-6).eq(&test_seed.mi, &hash.get("mi").unwrap()));

        // yields the same results as our python hit counting.
        let fname = "../../test_data/hits.npy";
        let hits_true: Array2<i64> = ndarray_npy::read_npy(fname).unwrap();
        assert_eq!(hits_true.sum(), hits.sum());
        assert_eq!(hits_true, hits);

        // get hits for the very first seed
        let other_seed = &mut seeds.seeds[0];
        let hits2 = rec_db.get_hits(
            &other_seed.params.params,
            &seeds.weights.weights_norm.view(),
            &threshold,
            &max_count,
        );
        
        let hit_cats2 = info_theory::categorize_hits(&hits2, &max_count);
        let h2v = hit_cats2.view();
        let contingency_3d = info_theory::construct_3d_contingency(
            hv,
            vv,
            h2v,
        );
        let cmi = info_theory::conditional_adjusted_mutual_information(
            contingency_3d.view()
        );

        assert!(AbsDiff::default().epsilon(1e-6).eq(&ami, &hash.get("mi").unwrap()));
        assert!(AbsDiff::default().epsilon(1e-6).eq(&cmi, &hash.get("cmi").unwrap()));
    }

    #[test]
    fn test_all_seeds () {
        // simulates args as they'll come from env::args in main.rs
        let args = [
            String::from("motifer"),
            String::from("../../test_data/subset_five_records.npy"),
            String::from("../../test_data/subset_five_y_vals.npy"),
            String::from("../../test_data/config.pkl"),
            String::from("../../test_data/test_output.pkl"),
        ];
        let cfg = parse_config(&args);
        let rec_db = RecordsDB::new_from_files(
            cfg.shape_fname,
            cfg.yvals_fname,
        );

        // create Seeds struct
        let mut seeds = rec_db.make_seed_vec(cfg.kmer, cfg.alpha);
        assert_eq!(seeds.seeds[0].mi, 0.0);

        seeds.compute_mi_values(
            &rec_db,
            &cfg.threshold,
            &cfg.max_count,
        );

        // test the first seed's MI has changed from its initial val of 0.0
        assert_ne!(seeds.seeds[0].mi, 0.0);

        let motifs = filter_seeds(
            &mut seeds,
            &rec_db,
            &cfg.threshold,
            &cfg.max_count,
        );
        println!("{:?}", motifs);
    }

    #[test]
    fn test_init_threshold () {
        // simulates args as they'll come from env::args in main.rs
        let args = [
            String::from("motifer"),
            String::from("../../test_data/shapes.npy"),
            String::from("../../test_data/y_vals.npy"),
            String::from("../../test_data/config.pkl"),
            String::from("../../test_data/test_output.pkl"),
        ];
        let cfg = parse_config(&args);
        let rec_db = RecordsDB::new_from_files(
            cfg.shape_fname,
            cfg.yvals_fname,
        );

        // create Seeds struct
        let mut seeds = rec_db.make_seed_vec(cfg.kmer, cfg.alpha);
        let thresh = set_initial_threshold(
            &seeds,
            &rec_db,
            cfg.seed_sample_size,
            cfg.records_per_seed,
            cfg.windows_per_record,
            &cfg.kmer,
            &cfg.alpha,
            cfg.thresh_sd_from_mean,
        );
        // assert that the python and rust estimates are within 0.03 of each other
        assert!(
            AbsDiff::default().epsilon(0.03).eq(&thresh, &cfg.threshold)
        );
        println!("Rust initial threshold: {}", thresh);
        println!("Python initial threshold: {}", cfg.threshold);
    }

    #[test]
    fn test_pickle_motifs() {
        let args = [
            String::from("motifer"),
            String::from("../../test_data/shapes.npy"),
            String::from("../../test_data/y_vals.npy"),
            String::from("../../test_data/config.pkl"),
            String::from("../../test_data/test_output.pkl"),
        ];
        let cfg = parse_config(&args);
 
        let length = 12;
        let this_seq = set_up_sequence(2.0, length);
        let that_seq = set_up_sequence(0.0, length);

        let mut this_motif = Motif::new(
            this_seq,
            1.0,
            10,
        );
        let mut that_motif = Motif::new(
            that_seq,
            1.0,
            10,
        );

        let motif_vec = vec![this_motif, that_motif];
        pickle_motifs(&motif_vec, &cfg.out_fname)
    }

}

/// Simple struct to hold command line arguments
pub struct Config {
    pub out_fname: String,
    pub shape_fname: String,
    pub yvals_fname: String,
    pub kmer: usize,
    pub alpha: f64,
    pub max_count: i64,
    pub threshold: f64,
    pub cores: usize,
    pub seed_sample_size: usize,
    pub records_per_seed: usize,
    pub windows_per_record: usize,
    pub thresh_sd_from_mean: f64,
}

impl Config {
    /// Returns a Config struct containing options contained in
    /// command line arguments
    ///
    /// # Arguments
    ///
    /// * `args` - an array of [String] structs. Comes from env::args in main.rs
    ///
    /// STILL NEEDS GOOD ERROR HANDLING, IMO
    pub fn new(args: &[String]) -> Config {
        let shape_fname = args[1].clone();
        let yvals_fname = args[2].clone();
        let opts_fname = args[3].clone();
        let out_fname = args[4].clone();

        // read in options we'll need
        let mut file = fs::File::open(opts_fname).unwrap();
        // open a buffered reader to open the pickle file
        let mut buf_reader = BufReader::new(file);
        // create a hashmap from the pickle file's contents
        let hash: HashMap<String, f64> = de::from_reader(
            buf_reader,
            de::DeOptions::new()
        ).unwrap();
        
        let kmer = *hash.get("kmer").unwrap_or(&15.0) as usize;
        let alpha = *hash.get("alpha").unwrap_or(&0.01) as f64;
        let max_count = *hash.get("max_count").unwrap_or(&1.0) as i64;
        let threshold = *hash.get("threshold").unwrap_or(&0.5) as f64;
        let cores = *hash.get("cores").unwrap_or(&1.0) as usize;
        let seed_sample_size = *hash.get("seed_sample_size")
            .unwrap_or(&250.0) as usize;
        let records_per_seed = *hash.get("records_per_seed")
            .unwrap_or(&50.0) as usize;
        let windows_per_record = *hash.get("windows_per_record")
            .unwrap_or(&1.0) as usize;
        let thresh_sd_from_mean = *hash.get("thresh_sd_from_mean")
            .unwrap_or(&2.0) as f64;

        Config{
            out_fname,
            shape_fname,
            yvals_fname,
            kmer,
            alpha,
            max_count,
            threshold,
            cores,
            seed_sample_size,
            records_per_seed,
            windows_per_record,
            thresh_sd_from_mean,
        }
    }
}

/// Parses arguments passed at the command line and places them into a [Config]
pub fn parse_config(args: &[String]) -> Config {
    Config::new(args)
}

/// Writes a vector of Motif structs as a pickle file
pub fn pickle_motifs(motifs: &Vec<Motif>, pickle_fname: &str) {
    // set up writer
    let mut file = fs::File::create(pickle_fname).unwrap();
    // open a buffered writer to open the pickle file
    let mut buf_writer = BufWriter::new(file);
    // write to the writer
    let res = ser::to_writer(
        &mut buf_writer, 
        motifs, 
        ser::SerOptions::new(),
    );
}

/// Filters seeds based on conditional mutual information
pub fn filter_seeds<'a>(seeds: &mut Seeds<'a>, rec_db: &'a RecordsDB, threshold: &f64, max_count: &i64) -> Vec<Motif> {

    // get number of parameters in model (shape number * seed length * 2)
    //  we multiply by two because we'll be optimizing shape AND weight values
    //  then add one for the threshold
    let rec_num = rec_db.len();
    let delta_k = seeds.seeds[0].params.params.raw_dim()[1]
        * seeds.seeds[0].params.params.raw_dim()[0]
        * 2 + 1;

    seeds.sort_seeds();
    let mut top_motifs = Vec::new();

    // Make sure first seed passes AIC
    let aic = info_theory::calc_aic(delta_k, rec_num, seeds.seeds[0].mi);
    if aic < 0.0 {
        let motif = seeds.seeds[0].to_motif(threshold);
        top_motifs.push(motif);
    } else {
        return top_motifs
    }

    // loop through candidate seeds
    for cand_seed in seeds.seeds[1..seeds.seeds.len()-1].iter() {

        let cand_hits = &cand_seed.hits;
        let cand_cats = info_theory::categorize_hits(&cand_hits, max_count);
        //let cc_view = cand_cats.view();
        let mut seed_pass = true;

        // if this seed doesn't pass AIC skip it
        if info_theory::calc_aic(delta_k, rec_num, cand_seed.mi) > 0.0 {
            continue
        }

        for good_motif in top_motifs.iter() {

            // check the conditional mutual information for this seed with
            //   each of the chosen seeds
            let good_motif_hits = &good_motif.hits;
            let good_cats = info_theory::categorize_hits(&good_motif_hits, max_count);
            //let gc_view = good_cats.view();

            let contingency = info_theory::construct_3d_contingency(
                cand_cats.view(),
                rec_db.values.view(),
                good_cats.view(),
            );
            let cmi = info_theory::conditional_adjusted_mutual_information(
                contingency.view()
            );
            let this_aic = info_theory::calc_aic(delta_k, rec_num, cmi);

            // if candidate seed doesn't improve model as added to each of the
            //   chosen seeds, skip it
            if this_aic > 0.0 {
                seed_pass = false;
                break
            }
        }
        if seed_pass {
            let motif = cand_seed.to_motif(threshold);
            top_motifs.push(motif);
        }
    }
    return top_motifs
}

/// An enumeration of possible parameter names
/// Supported parameters could be added here in the future
/// We could also generalize it to be just a String type to take
/// anyones name for a parameter
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum ParamType {
    EP,
    MGW,
    Roll,
    ProT,
    HelT,
}


/// Container struct for a single parameter vector.
///
/// # Fields
///
/// * `name` - An associated name of type [ParamType]
/// * `vals` - Values are stored as a 1 dimensional ndarray of floating point numbers
#[derive(Debug)]
pub struct Param {
    name: ParamType,
    vals: ndarray::Array::<f64,Ix1>,
}

/// Represents a single sequence as a combination of [Param] objects
///
/// # Fields
///
/// * `params` - Stores the full set of params in a single 2d Array
#[derive(Debug, Serialize, Deserialize)]
pub struct Sequence {
    params: ndarray::Array2<f64>
}

/// Represents a single stranded sequence as a combination of [Param] objects
///
/// # Fields
///
/// * `params` - Stores the full set of params in a single 3d Array
#[derive(Debug)]
pub struct StrandedSequence {
    params: ndarray::Array3<f64>
}

/// Represents the state needed for windowed iteration over a [StrandedSequence]
///
/// # Fields
///
/// * `start` - start position of the iteration
/// * `end` - exclusive end of the iteration
/// * `size` - size of the window to iterate over
/// * `sequence` - reference to the [StrandedSequence] to iterate over
#[derive(Debug)]
pub struct StrandedSequenceIter<'a>{
    start: usize,
    end: usize,
    size: usize,
    sequence: &'a StrandedSequence
}

/// Represents the state needed for random,
/// windowed iteration over a [StrandedSequence]
///
/// # Fields
///
/// * `start` - start position of the iteration
/// * `end` - exclusive end of the iteration
/// * `size` - size of the window to iterate over
/// * `sequence` - reference to the [StrandedSequence] to iterate over
/// * `indices` - the randomized indices that will be iterated over
#[derive(Debug)]
pub struct PermutedStrandedSequenceIter<'a>{
    start: usize,
    end: usize,
    size: usize,
    sample_size: usize,
    sequence: &'a StrandedSequence,
    indices: Vec<usize>,
}

/// Represents the state needed for windowed iteration over a [StrandedSequence],
/// but to return only the forward strand.
///
/// # Fields
///
/// * `start` - start position of the iteration
/// * `end` - exclusive end of the iteration
/// * `size` - size of the window to iterate over
/// * `sequence` - reference to the [StrandedSequence] to iterate over
#[derive(Debug)]
pub struct FwdStrandedSequenceIter<'a>{
    start: usize,
    end: usize,
    size: usize,
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    // We can probably save a lot by making this a StrandedSequenceView, but I honestly haven't thought about it in a while.
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    sequence: &'a StrandedSequence
}

/// Represents an immutable windowed view to a [StrandedSequence]
///
/// # Fields
///
/// * `params` - The view is stored as a 3d ndarray
#[derive(Debug)]
pub struct StrandedSequenceView<'a> {
    params: ndarray::ArrayView::<'a, f64, Ix3>
}

/// Represents the state needed for windowed iteration over a [Sequence]
///
/// # Fields
///
/// * `start` - start position of the iteration
/// * `end` - exclusive end of the iteration
/// * `size` - size of the window to iterate over
/// * `sequence` - reference to the [Sequence] to iterate over
#[derive(Debug)]
pub struct SequenceIter<'a>{
    start: usize,
    end: usize,
    size: usize,
    sequence: &'a Sequence
}


/// Represents an immutable windowed view to a [Sequence]
///
/// # Fields
///
/// * `params` - The view is stored as a 2d ndarray
#[derive(Debug, Copy, Clone)]
pub struct SequenceView<'a> {
    params: ndarray::ArrayView::<'a, f64, Ix2>
}

/// Represents a motif as a [SequenceView] with associated [MotifWeights]
///
/// # Fields
///
/// * `params` - Stores the sequence values as a [SequenceView]
/// * `weights` - Stores the associated weights as a [MotifWeights]
/// * `threshold` - Stores the associated threshold for what is
///                 considered a match.
#[derive(Debug, Serialize, Deserialize)]
pub struct Motif {
    params: Sequence,
    weights: MotifWeights, // the MotifWeights struct contains two 2D-Arrays.
    threshold: f64,
    hits: ndarray::Array2::<i64>,
    mi: f64,
}

/// Represents the weights for a [Motif] in it's own structure
///
/// # Fields
///
/// * `weights` - Stores the weights as a 2d array
/// * `weights_norm` - Caches normalized weights as needed
#[derive(Debug, Serialize, Deserialize)]
pub struct MotifWeights {
    weights: ndarray::Array2::<f64>,
    weights_norm: ndarray::Array2::<f64>,
}

// My goal for this struct is to be able to have a way to use a read-only pointer
//  to each window of sequences using a SequenceView, and to have all Seeds point
//  to the same set of weights using the ArrayView.
// This way we can calculate all our initial MIs without copying unnecessarily.
// After filtering by CMI we can then create Motif structs that each own their
//  shapes and weights. 

// We'll need to do something to serialize either the [Seeds] struct,
//  the [Seed] struct, or both, to be able to rapidly pass seeds back to python.
// Uncommenting the next line just yield a bunch of not implemented errors.
//#[derive(Debug, Serialize, Deserialize)]
#[derive(Debug)]
pub struct Seed<'a> {
    params: SequenceView<'a>,
    hits: ndarray::Array2::<i64>,
    mi: f64,
}

// We have to create a container to hold the weights with the seeds
#[derive(Debug)]
pub struct Seeds<'a> {
    seeds: Vec<Seed<'a>>,
    weights: MotifWeights
}

/// Represents a database of Sequences and their associated value
///
/// # Fields
///
/// * `seqs` - Stores [Sequence] classes in a vector
/// * `values` - Stores associated values in a vector in 1D array
#[derive(Debug)]
pub struct RecordsDB {
    seqs: Vec<StrandedSequence>,
    values: ndarray::Array1::<i64>
}


/// Allows for iteration over a records database
///
/// # Fields
///
/// * `loc` - Current location in the database
/// * `value` - A reference to the [RecordsDB]
#[derive(Debug)]
pub struct RecordsDBIter<'a> {
    loc: usize,
    db: &'a RecordsDB,
    size: usize
}

/// Allows for iteration over a permuted records database
///
/// # Fields
///
/// * `loc` - Current location in the permuted database
/// * `value` - A reference to the [RecordsDB]
/// * `idx` - Index to grab for a given iteration over the permuted database
/// * `idx` - Index to grab for a given iteration over the permuted database
#[derive(Debug)]
pub struct PermutedRecordsDBIter<'a> {
    loc: usize,
    db: &'a RecordsDB,
    sample_size: usize,
    indices: Vec<usize>,
}

/// Stores a single entry of the RecordsDB 
/// # Fields
///
/// * `seq` - A reference to a [Sequence] classe
/// * `value` - The associated value for the sequence
#[derive(Debug)]
pub struct RecordsDBEntry<'a> {
    seq: &'a StrandedSequence,
    value: i64
}

impl StrandedSequence {
    /// Returns a Result containing a new stranded sequence or any errors that 
    /// occur in attempting to create it.
    ///
    /// # Arguments
    ///
    /// * `array` - a 3D array of shapes
    ///
    /// This is volatile code and likely to change based on how we read
    /// in the initial parameters
    pub fn new(array: ndarray::Array3<f64>) -> StrandedSequence {
        StrandedSequence{ params: array }
    }

    /// Creates a read-only windowed iterator over the sequence. Automatically
    /// slides by 1 unit.
    ///
    /// # Arguments
    ///
    /// * `start` - the starting position in the sequence to begin iteration
    /// * `end` - the ending position in the sequence to stop iteration. End is excluded
    /// * `size` - the size of the window to slide over
    pub fn window_iter(
        &self, start: usize, end: usize, size: usize
    ) -> StrandedSequenceIter {
        StrandedSequenceIter{start, end, size, sequence: self}
    }

    pub fn fwd_strand_window_iter(
        &self, start: usize, end: usize, size: usize
    ) -> FwdStrandedSequenceIter {
        FwdStrandedSequenceIter{start, end, size, sequence: self}
    }

    pub fn random_window_iter(
        &self, start: usize, end: usize, size: usize, sample_size: usize
    ) -> PermutedStrandedSequenceIter {
        // create vector of indices
        let mut indices: Vec<usize> = (0..self.seq_len()-size).collect();
        // randomly shuffle the indices
        indices.shuffle(&mut thread_rng());
        if sample_size > self.seq_len()-size {
            let sample_size = self.seq_len()-size;
        }
        PermutedStrandedSequenceIter{
            start,
            end,
            size,
            sample_size,
            sequence: self,
            indices,
        }
    }

    /// Returns a read-only StrandedSequenceView pointing to the data in Sequence
    pub fn view(&self) -> StrandedSequenceView {
        StrandedSequenceView::new(self.params.view())
    }

    pub fn seq_len(&self) -> usize {
        self.params.raw_dim()[1]
    }
    pub fn param_num(&self) -> usize{
        self.params.raw_dim()[0]
    }

    /// For a single Motif, count the number of times its distance to a window
    /// of a Sequence falls below the specified threshold, i.e., matches the
    /// Sequence.
    ///
    /// # Arguments
    ///
    /// * `query` - an array which will be compared to each window in self.
    /// * `weights` - an array of weights to be applied for the distance calc
    /// * `threshold` - distance between the query and seq below which a hit is called
    /// * `max_count` - maximum number of times a hit will be counted on each strand
    pub fn count_hits_in_seq(&self, query: &ndarray::ArrayView<f64,Ix2>,
                             weights: &ndarray::ArrayView<f64, Ix2>,
                             threshold: &f64,
                             max_count: &i64) -> Array<i64, Ix1> {
    
        // set maxed to false for each strand
        let mut f_maxed = false;
        let mut r_maxed = false;
        let mut hits = ndarray::Array::zeros(2);
    
        // iterate through windows of seq
        for window in self.window_iter(0, self.seq_len()+1, query.raw_dim()[1]) {
    
            // once both strands are maxed out, stop doing comparisons
            if f_maxed & r_maxed {
                break
            }
            // get the distances.
            let dist = stranded_weighted_manhattan_distance(
                &window.params,
                query,
                weights,
            );

            if (dist[0] < *threshold) & (!f_maxed) {
                hits[0] += 1;
                if hits[0] == *max_count {
                    f_maxed = true;
                }
            } 

            if (dist[1] < *threshold) & (!r_maxed) {
                hits[1] += 1;
                if hits[1] == *max_count {
                    r_maxed = true;
                }
            } 
    
        }
        // return the hits
        hits
    }
}

impl Sequence {
    /// Returns a Result containing a new sequence or any errors that 
    /// occur in attempting to create it.
    ///
    /// # Arguments
    ///
    /// * `params` - A vector of [Param] objects
    ///
    /// This is volatile code and likely to change based on how we read
    /// in the initial parameters
    pub fn new(params: Vec<Param>) -> Result<Sequence, Box<dyn Error>> {
        // figure out how many rows and columns we will need
        let nrows = params.len();
        let ncols = params[0].vals.len();
        // Allocate an array to store the whole sequence
        let mut arr = ndarray::Array2::zeros((nrows, ncols));
        // iterate over the row axis pull each row (from ndarray docs)
        for (i, mut row) in arr.axis_iter_mut(Axis(0)).enumerate() {
            // get a pointer to the inner array for the param
            let this_param = &params[i].vals;
            // copy the data into the new array
            row.assign(this_param);
        }
        Ok(Sequence{ params: arr })
    }

    /// Creates a read-only windowed iterator over the sequence. Automatically
    /// slides by 1 unit.
    ///
    /// # Arguments
    ///
    /// * `start` - the starting position in the sequence to begin iteration
    /// * `end` - the ending position in the sequence to stop iteration. End is excluded
    /// * `size` - the size of the window to slide over
    pub fn window_iter(&self, start: usize, end: usize, size: usize) -> SequenceIter {
        SequenceIter{start, end, size, sequence: self}
    }

    // Insert method here to do the comparisons. this way 

    /// Returns a read-only SequenceView pointing to the data in Sequence
    pub fn view(&self) -> SequenceView {
        SequenceView::new(self.params.view())
    }

    pub fn seq_len(&self) -> usize {
        self.params.raw_dim()[1]
    }
    pub fn param_num(&self) -> usize{
        self.params.raw_dim()[0]
    }

    /// For a single Motif, count the number of times its distance to a window
    /// of a Sequence falls below the specified threshold, i.e., matches the
    /// Sequence.
    ///
    /// # Arguments
    ///
    /// * `query` - an array which will be compared to each window in seq.
    /// * `weights` - an array of weights to be applied for the distance calc
    /// * `threshold` - distance between the query and seq below which a hit is called
    /// * `max_count` - maximum number of times a hit will be counted on each strand
    pub fn count_hits_in_seq(&self, query: &ndarray::ArrayView<f64,Ix2>,
                             weights: &ndarray::ArrayView<f64, Ix2>,
                             threshold: &f64, max_count: &i64) -> Array<i64, Ix1> {
    
        // set maxed to false for each strand
        let mut f_maxed = false;
        //////////////////////////////////
        // SET TO TRUE FOR REVERSE UNTIL WE ACTUALLY START USING STRANDEDNESS
        //////////////////////////////////
        let mut r_maxed = true;
        let mut hits = ndarray::Array::zeros(2);
    
        // iterate through windows of seq
        for window in self.window_iter(0, self.seq_len()+1, query.raw_dim()[1]) {
    
            // once both strands are maxed out, stop doing comparisons
            if f_maxed & r_maxed {
                break
            }
            // get the distance.
            /////////////////////////////////////////////////
            // IN THE FUTURE, WE'LL BROADCAST TO BOTH STRANDS
            /////////////////////////////////////////////////
            let dist = weighted_manhattan_distance(&window.params,
                                                   query,
                                                   weights);
            /////////////////////////////////////////////////
            // ONCE WE'VE IMPLEMENTED STRANDEDNESS, SLICE APPROPRIATE STRAND'S DISTANCE HERE
            /////////////////////////////////////////////////
            if (dist < *threshold) & (!f_maxed) {
                hits[0] += 1;
                if hits[0] == *max_count {
                    f_maxed = true;
                }
            } 
            /////////////////////////////////////////////////
            /////////////////////////////////////////////////
            /////////////////////////////////////////////////
            //if (dist[1] < threshold) & (!r_maxed) {
            //    hits[1] += 1;
            //    if hits[1] == max_count {
            //        r_maxed = true;
            //    }
            //} 
    
        }
        // return the hits
        hits
    }

}

impl<'a> StrandedSequenceView<'a> {
    /// Creates a immutable view from a subset of a [Sequence]
    ///
    /// # Arguments
    ///
    /// * `params` - a vector of ndarray slices representing a subset of the given sequence
    pub fn new(params: ndarray::ArrayView::<'a,f64, Ix3>) -> StrandedSequenceView<'a> {
        StrandedSequenceView { params }
    }
    
    /// Returns an iterator over the views of each [Param]
    pub fn iter(&self) -> ndarray::iter::AxisIter<f64, Ix2>{
        self.params.axis_iter(Axis(2))
    }
}

impl<'a> SequenceView<'a> {
    /// Creates a immutable view from a subset of a [Sequence]
    ///
    /// # Arguments
    ///
    /// * `params` - a vector of ndarray slices representing a subset of the given sequence
    pub fn new(params: ndarray::ArrayView::<'a,f64, Ix2>) -> SequenceView<'a> {
        SequenceView { params }
    }
    
    /// Returns an iterator over the views of each [Param]
    pub fn iter(&self) -> ndarray::iter::AxisIter<f64, Ix1>{
        self.params.axis_iter(Axis(0))
    }
}

/// Enables iteration over a given sequence. Returns a [SequenceView] at each
/// iteration
impl<'a> Iterator for FwdStrandedSequenceIter<'a> {
    type Item = SequenceView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let this_start = self.start;
        let this_end = self.start + self.size;
        if this_end == self.end {
            None
        } else {
            let out = self.sequence.params.slice(s![..,this_start..this_end,0]);
            self.start += 1;
            Some(SequenceView::new(out))
        }
    }
}

/// Enables iteration over a given StrandedSequence.
/// Returns a [StrandedSequenceView] at each iteration.
impl<'a> Iterator for StrandedSequenceIter<'a> {
    type Item = StrandedSequenceView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let this_start = self.start;
        let this_end = self.start + self.size;
        if this_end == self.end {
            None
        } else {
            let out = self.sequence.params.slice(s![..,this_start..this_end,..]);
            self.start += 1;
            Some(StrandedSequenceView::new(out))
        }
    }
}

/// Enables random iteration over a given StrandedSequence.
/// Returns a [StrandedSequenceView] at each iteration.
impl<'a> Iterator for PermutedStrandedSequenceIter<'a> {
    type Item = StrandedSequenceView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        // start at the 0-th index of the randomly shuffled indices in self.indices
        let this_start = self.indices[self.start];
        // add the window width, self.size, to the start to get the end position
        let this_end = this_start + self.size;
        //println!("This start: {}\nThis end: {}", this_start, this_end);
        // if we've reached the sample size, return None, exiting the iterator
        if self.start == self.sample_size {
            None
        } else {
            let out = self.sequence.params.slice(s![..,this_start..this_end,..]);
            self.start += 1;
            Some(StrandedSequenceView::new(out))
        }
    }
}

/// Enables iteration over a given sequence. Returns a [SequenceView] at each
/// iteration
impl<'a> Iterator for SequenceIter<'a> {
    type Item = SequenceView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let this_start = self.start;
        let this_end = self.start + self.size;
        if this_end == self.end {
            None
        } else {
            let out = self.sequence.params.slice(s![..,this_start..this_end]);
            self.start += 1;
            Some(SequenceView::new(out))
        }
    }
}


impl Param {

    /// Returns a new container for a single parameter
    ///
    /// # Arguments
    ///
    /// * `name` - a type for the parameter. Must be one of [ParamType]
    /// * `vals` - an 1 dimensional array of floating point values for the parameter
    pub fn new(name: ParamType, vals: Array::<f64, Ix1>) -> Result<Param, Box<dyn Error>> {
        Ok(Param {name, vals})
    }

    /// Enables subtraction between two full parameter vectors without
    /// direct access to the inner values.
    ///
    /// Returns the element-wise subtraction.
    pub fn subtract(&self, other: &Param) -> Vec<f64> {
        if self.name != other.name {
            panic!("Can't subtract params of different types")
        } else {
            self.iter().zip(other).map(|(a, b)| a - b).collect()
        }
    }

    /// Returns an iterator over the individual values in the inner array
    fn iter(&self) -> ndarray::iter::Iter<f64, Ix1> {
        self.vals.iter()
    }

    /// Returns a windowed iterator over the inner values array
    ///
    /// # Arguments
    ///
    /// * `size` - size of the window to iterate over
    fn windows(&self, size: usize) -> ndarray::iter::Windows<f64, Ix1>{
        self.vals.windows(size)
    }
}

/// This allows the syntactic sugar of `for val in param` to work
impl<'a> IntoIterator for &'a Param {
    type Item = &'a f64;
    type IntoIter = ndarray::iter::Iter<'a, f64, Ix1>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}


impl Motif {
    /// Returns a new motif instance by bundling with a weight vector
    /// of [MotifWeights] type. 
    ///
    /// # Arguments
    ///
    /// * `params` - a [Sequence] that defines the motif
    pub fn new(params: Sequence, threshold: f64, record_num: usize) -> Motif {
        let weights = MotifWeights::new(&params.view());
        let mut hits = ndarray::Array2::zeros((record_num, 2));
        let mi = 0.0;
        Motif{params, weights, threshold, hits, mi}
    }

    /// Does constrained normalization of weights
    ///
    /// # Arguments
    ///
    /// * `alpha` - lower limit on weights after inv-logit transform,
    ///   but prior to their normalization to sum to one.
    pub fn normalize_weights(&mut self, alpha: &f64) -> () {
        self.weights.constrain_normalize(alpha);
    }

    /// Calculates distance between this motif and each strand of
    /// a reference SequenceView (well, it will do it for each strand...
    /// for now we are only implementing the forward strand).
    ///
    /// # Arguments
    ///
    /// * `ref_seq` - a [SequenceView] to compare this motif against.
    pub fn distance(&self, ref_seq: &SequenceView) -> f64 {
        weighted_manhattan_distance(
            &self.params.params.view(), 
            &ref_seq.params, 
            &self.weights.weights_norm.view()
        )
    }
}


impl<'a> Seed<'a> {

    pub fn new(params: SequenceView<'a>,
               record_num: usize) -> Seed<'a> {
        let hits = ndarray::Array2::zeros((record_num, 2));
        let mi = 0.0;
        Seed{params, hits, mi}
    }
    
    fn update_hits(&mut self, db: &RecordsDB,
                       weights: &ndarray::ArrayView<f64, Ix2>,
                       threshold: &f64,
                       max_count: &i64) {
        self.hits = db.get_hits(
            &self.params.params,
            weights,
            threshold,
            max_count,
        )
    }

    fn update_mi(&mut self, db: &RecordsDB, max_count: &i64) {
        let hit_cats = info_theory::categorize_hits(&self.hits, &max_count);
        let hv = hit_cats.view();
        let vv = db.values.view();
        let contingency = info_theory::construct_contingency_matrix(hv, vv);
        self.mi = info_theory::adjusted_mutual_information(contingency.view())
    }

    pub fn update_info(&mut self, db: &RecordsDB,
                       weights: &ndarray::ArrayView<f64, Ix2>,
                       threshold: &f64,
                       max_count: &i64) {
        self.update_hits(
            db,
            weights,
            threshold,
            max_count,
        );
        self.update_mi(
            db,
            max_count,
        );
    }

    /// Returns a new motif from a seed.
    pub fn to_motif(&self,
            threshold: &f64) -> Motif {
        let weights = MotifWeights::new(&self.params);
        // I think I may just need to copy the hits and params
        //  to make the motif own them.
        let hits = self.hits.to_owned();
        let arr = self.params.params.to_owned();
        let params = Sequence{ params: arr };
        let mi = self.mi;
        Motif{params, weights, threshold: *threshold, hits, mi}
    }

}


///// Allow conversion from a [SequenceView] to a [Motif]
//impl<'a> From<SequenceView<'a>> for Motif<'a> {
//    fn from(sv : SequenceView<'a>, rec_num: usize) -> Motif<'a>{
//        Motif::new(sv, 0.0, rec_num)
//    }
//}


impl<'a> MotifWeights {
    /// Returns a new motif weight instance based on the size of a
    /// [SequenceView]. Initializes all weights with one.
    /// Initializes a weights_norm array to all zeros.
    ///
    /// # Arguments
    ///
    /// * `params` - a [SequenceView] used strictly to define the 
    ///              size of the weights
    pub fn new(params: &SequenceView) -> MotifWeights {
        let weights = Array::<f64, Ix2>::ones(params.params.raw_dim());
        let weights_norm = Array::<f64, Ix2>::zeros(params.params.raw_dim());
        MotifWeights{ weights, weights_norm }
    }

    pub fn new_bysize(rows: usize, cols: usize) -> MotifWeights {
        let weights = Array::<f64, Ix2>::ones((rows, cols));
        let weights_norm = Array::<f64, Ix2>::zeros((rows, cols));
        MotifWeights{ weights, weights_norm}
    }
    
    /// Updates the weights_norm field in place based on the weights field.
    /// The motif must be mutably borrowed to be able to use this method
    ///
    /// Weights are normalized as exp(weight)/sum(exp(weights))
    pub fn normalize(&mut self) {
        let total = ndarray::Zip::from(&self.weights)
            .fold(0.0, |acc, a| acc + f64::exp(*a));
        // trying to do this in place consuming the old values
        ndarray::Zip::from(&mut self.weights_norm)
            .and(&self.weights).for_each(|a, b| *a = f64::exp(*b)/total);
    }

    /// Updates the weights_norm field in place based on the weights field.
    /// 
    /// Weights have the constrained inv_logit function applied to them, then are
    ///  normalized to sum to one.
    pub fn constrain_normalize(&mut self, alpha: &f64) {
        let total = ndarray::Zip::from(&self.weights)
            .fold(0.0, |acc, a| acc + inv_logit(*a, Some(*alpha)));
        // deref a and b here to modify values in place
        ndarray::Zip::from(&mut self.weights_norm)
            .and(&self.weights)
            .for_each(|a, b| *a = inv_logit(*b, Some(*alpha))/total);
    }
}

impl<'a> Seeds<'a> {

    /// Return the length of the vector of seeds in Seeds
    pub fn len(&self) -> usize {
        self.seeds.len()
    }

    /// iterates over seeds, getting hits and mi, updating the Seed
    /// structs' mi and hits values as it goes
    ///
    /// # Arguments
    ///
    /// * `rec_db` - A reference to a [RecordsDB] against which
    ///     each [Seed] in self will be compared.
    pub fn compute_mi_values(&mut self,
                             rec_db: &RecordsDB,
                             threshold: &f64,
                             max_count: &i64) {

        // grab a view to the normalized weights. Cannot be done within the loop,
        // since that would be borrowing a reference to self twice in the iterator.
        let weights_view = self.weights.weights_norm.view();
        // iterate over vector of [Seed]s in parallel, updating hits and mi vals
        self.seeds.par_iter_mut().for_each(|seed| {
            // update_info does hit counting and ami calculation, placing
            // the results into seed.hits and seed.mi
            seed.update_info(
                rec_db,
                &weights_view,
                threshold,
                max_count,
            );
        });
    }

    /// Sorts seeds by mi
    pub fn sort_seeds(&mut self) {
        self.seeds.par_sort_unstable_by_key(|seed| OrderedFloat(-seed.mi));
    }

}

impl RecordsDB {

    /// Returns a new RecordsDB holding sequence value pairs as separate
    /// vectors.
    ///
    /// # Arguments
    ///
    /// * `seqs` - a vector of [Sequence]
    /// * `values` - a vector of values for each sequence
    pub fn new(seqs: Vec<StrandedSequence>, values: ndarray::Array1::<i64>) -> RecordsDB {
        RecordsDB{seqs, values}
    }

    /// Reads in input files to return a RecordsDB
    ///
    /// # Arguments
    ///
    /// * `shape_file` - a npy file containing a 4D array of shape (R,L,S,2),
    ///      where R is the record number, L is the length of each sequence,
    ///      S is the number of shape parameters, and the final axis is of
    ///      length 2 to hold shape information for each of the two strands.
    /// * `y_val_file` - a npy file containing a 1D array of shape (R). Each
    ///      element of the array contains the given record's category.
    pub fn new_from_files(shape_file: String, y_val_file: String) -> RecordsDB {
        // read in the shape values and permute axes so that,
        //   instead of being shape (R,L,S,strand), arr is of
        //   shape (R,S,L,strand)
        let arr: Array4<f64> = ndarray_npy::read_npy(shape_file).unwrap();
        //let arr = input.permuted_axes([0,2,1,3]);
        
        // iterate over records, creating StrandedSequnce structs from them,
        //   and push them into a vector
        let mut seq_vec = Vec::new();
        for r in 0..arr.raw_dim()[0] {
            let seq_arr = arr.slice(s![r,..,..,..]).to_owned();
            let seq = StrandedSequence::new(seq_arr);
            seq_vec.push(seq);
        }

        // read in the categories for each record
        let y_vals: Array1<i64> = ndarray_npy::read_npy(y_val_file).unwrap();
        RecordsDB::new(seq_vec, y_vals)
    }

    /// Returns the number of records in RecordsDB
    pub fn len(&self) -> usize {
        self.seqs.len()
    }

    /// Return a vector of Seed structs.
    ///
    /// # Arguments
    ///
    /// * `kmer` - Length of windows to slice over each record's parameters
    /// * `alpha` - Lower limit on inv_logit transformed weights
    pub fn make_seed_vec(&self, kmer: usize, alpha: f64) -> Seeds {
        // want to fill a vector of possible seeds 
        let mut seed_vec = Vec::new();
        // we want them all to have the same kmer size over all parameters
        // How many parameters? All sequences should have the same number of
        // parameters stored in rows in their params vector
        let param_num = self.seqs[0].param_num();
        // The number of columns is simply the kmer size. We define an empty
        // sequence the size of a kmer
        let mut seed_weights = MotifWeights::new_bysize(param_num, kmer);
        seed_weights.constrain_normalize(&alpha);

        for entry in self.iter() {
            for window in entry.seq.fwd_strand_window_iter(0, entry.seq.seq_len()+1, kmer) {
                seed_vec.push(Seed::new(window, self.len()));
            }
        }
        // We can't store a reference to the seed weights in each seed since
        // the seed_weights is locally defined in this function and goes out
        // of scope when the function ends. Instead we will make a container
        // class to hold the seeds together with their shared weights and
        // pass full ownership of the weights out of the function.
        Seeds{seeds: seed_vec, weights: seed_weights}
    }

    /// Iterate over each record in the database as a [StrandedSequence] value pair
    pub fn iter(&self) -> RecordsDBIter{
        RecordsDBIter{loc: 0, db: &self, size: self.seqs.len()}
    }

    /// Permute the indices of the database, then iterate over the permuted indices
    pub fn random_iter(&self, sample_size: usize) -> PermutedRecordsDBIter{
        // create vector of indices
        let mut inds: Vec<usize> = (0..self.seqs.len()).collect();
        // randomly shuffle the indices
        inds.shuffle(&mut thread_rng());
        // return the struct for iterating
        let mut size = sample_size;
        if sample_size >= self.seqs.len() {
            size = self.seqs.len();
        }
        PermutedRecordsDBIter{
            loc: 0,
            db: &self,
            sample_size: size,
            indices: inds,
        }
    }


    /// Iterate over each record in the database and count number of times
    /// `query` matches each record. Return a 2D array of hits, where each
    /// row represents a record in the database, and each column is the number
    /// of hits counted on each strand for a given record.
    ///
    /// # Arguments
    ///
    /// * `query` - A 2D arrayview, typically coming from a Seed's or Motif's params
    /// * `weights` - A 2D arrayview, typically coming from a Motif or a Seeds struct
    /// * `threshold` - The threshold distance below which a hit is called
    /// * `max_count` - The maximum number of hits to call for each strand
    pub fn get_hits(
        &self,
        query: &ndarray::ArrayView<f64, Ix2>,
        weights: &ndarray::ArrayView<f64, Ix2>,
        threshold: &f64,
        max_count: &i64) -> Array<i64, Ix2> {

        let mut hits = ndarray::Array2::zeros((self.len(), 2));
        hits.axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(&self.seqs)
            .for_each(|(mut row, seq)| {
                let this_hit = seq.count_hits_in_seq(
                    query,
                    weights,
                    threshold,
                    max_count,
                );
                row.assign(&this_hit);
            });

//        //for (i, entry) in self.iter().enumerate() {
//            //let this_hit = entry.seq.count_hits_in_seq(
//            let this_hit = seq.count_hits_in_seq(
//                query,
//                weights,
//                threshold,
//                max_count,
//            );
//            //println!("{:?}", this_hit);
//            hits.row_mut(i).assign(&this_hit);
//        });

        sort_hits(&mut hits);

        hits
    }
}

impl<'a> RecordsDBEntry<'a> {
    /// Returns a single [RecordsDBEntry] holding the sequence value
    /// pair
    ///
    /// # Arguments
    /// * `seq` - a reference to a [Sequence]
    /// * `value` - the sequences paired value
    pub fn new(seq: &StrandedSequence, value: i64) -> RecordsDBEntry {
        RecordsDBEntry{seq, value}
    }
}

/// Enables iteration over the RecordsDB. Returns a [RecordsDBEntry] as 
/// each item.
impl<'a> Iterator for RecordsDBIter<'a> {
    type Item = RecordsDBEntry<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.loc == self.size{
            None
        } else {
            let out_seq = &self.db.seqs[self.loc];
            let out_val = self.db.values[self.loc];
            self.loc += 1;
            Some(RecordsDBEntry::new(out_seq, out_val))
        }
    }
}

///// Enables parallel iteration over the entries in RecordsDB.
///// Returns a [RecordsDBEntry] as each item.
//impl<'a> ParallelIterator for RecordsDBIter<'a> {
//    type Item = RecordsDBEntry<'a>;
//
//    fn drive_unindexed<C>(self, consumer: C) -> C::Result
//    where
//        C: UnindexedConsumer<Self::Item>,
//    {
//    
////////////////////////////////////////////////////
//    fn drive_unindexed<C>(self, consumer: C) -> C::Result
//    where
//        C: UnindexedConsumer<Self::Item>,
//    {
//        let StepBy5 { start, end } = self;
//
//        // Start with a simple range, then map it to increment by 5
//        (0..(end - start + 4) / 5)
//            .into_par_iter()
//            .map(|i| start + i * 5)
//            .drive_unindexed(consumer)
//    }
////////////////////////////////////////////////////
//

/// Enables permuted iteration over the RecordsDB. Returns a [RecordsDBEntry] as 
/// each item.
impl<'a> Iterator for PermutedRecordsDBIter<'a> {
    type Item = RecordsDBEntry<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.loc == self.sample_size{
            None
        } else {
            let out_seq = &self.db.seqs[self.indices[self.loc]];
            let out_val = self.db.values[self.indices[self.loc]];
            self.loc += 1;
            Some(RecordsDBEntry::new(out_seq, out_val))
        }
    }
}

/// Calculate distances for randomly chosen seeds and randomly selected
/// RecordsDBEntry structs
///
/// # Arguments
pub fn set_initial_threshold(seeds: &Seeds, rec_db: &RecordsDB, seed_sample_size: usize, records_per_seed: usize, windows_per_record: usize, kmer: &usize, alpha: &f64, thresh_sd_from_mean: f64) -> f64 {

    let seed_vec = &seeds.seeds;
    let mut inds: Vec<usize> = (0..seed_vec.len()).collect();
    inds.shuffle(&mut thread_rng());
    let keeper_inds = &inds[0..seed_sample_size];

    let rows = seed_vec[0].params.params.raw_dim()[0];
    let cols = seed_vec[0].params.params.raw_dim()[1];

    let mut mw = MotifWeights::new_bysize(rows, cols);
    mw.constrain_normalize(alpha);

    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////
    // NOTE: the random iter methods should really be tested to check their randomness
    // I think simply asserting that a sample of 4 yields 4 items that do not match
    // the first four items in the original sturct would be sufficient.
    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////

    let mut distances = Vec::new();
    for (i,seed) in seed_vec.iter().enumerate() {
        if keeper_inds.contains(&i) {
            for entry in rec_db.random_iter(records_per_seed) {
                let seq = entry.seq;
                for window in seq.random_window_iter(0, seq.seq_len()+1, *kmer, windows_per_record) {
                    // get the distances.
                    let dist = stranded_weighted_manhattan_distance(
                        &window.params,
                        &seed.params.params,
                        &mw.weights_norm.view(),
                    );
                    distances.push(dist[0]);
                    distances.push(dist[1]);
                }
            }
        }
    }

    // could do these sums in parallel, but we wouldn't get that much benefit
    let dist_sum = distances.iter().sum::<f64>();
    let mean_dist: f64 = dist_sum
        / distances.len() as f64;
    // let mean_dist: f64 = distances.iter().sum::<f64>() / distances.len() as f64;
    let variance: f64 = distances.iter()
        .map(|a| {
            let diff = a - mean_dist;
            diff * diff
        })
        .sum::<f64>()
        / distances.len() as f64;
    let std_dev = variance.sqrt();
    mean_dist - std_dev * thresh_sd_from_mean
}


/// Function to compute manhattan distance between two 2D array views
///
/// # Arguments
///
///
/// - `arr1` - a reference to a view of a 2D array, typically a window on a sequence
/// - `arr2` - a reference to a view of a 2D array, typically a window on a sequence to be compared
pub fn manhattan_distance(arr1: &ndarray::ArrayView::<f64, Ix2>, 
                          arr2: &ndarray::ArrayView::<f64, Ix2>) -> f64 {
    ndarray::Zip::from(arr1).
        and(arr2).
        fold(0.0, |acc, a, b| acc + (a-b).abs())
}

/// Function to compute a constrained manhattan distance between two 2D array 
/// views with a single associated set of weights. Views are used so that this
/// can eventually be parallelized if needed.
///
/// # Arguments
///
/// - `arr1` - a reference to a view of a 2D array, typically a window on a sequence to be compared
/// - `arr2` - a reference to a view of a 3D array, typically a [Motif] `param` field
/// - `weights` - a view of a 2D array, typically a [Motif] `weights` field
pub fn stranded_weighted_manhattan_distance(
    arr1: &ndarray::ArrayView::<f64, Ix3>, 
    arr2: &ndarray::ArrayView::<f64, Ix2>,
    weights: &ndarray::ArrayView::<f64, Ix2>,
) -> ndarray::Array1<f64> {

    ////////////////////////////////////////////////////
    // NOTE: here's a perfect target for a benchmark test. Run the commented version,
    //   and also run the uncommented version, check the speed of each. This will
    //   benefit us greatly, since we run this function tens-of-thousands,
    //   to hundreds-of-thousands of times.
    // leave this here for now. It yields identical results to the code below.
    // We could check to see which version is more performant.
    // I'm currently just guessing that the iter methods that are currently used are faster
    //   because they probably do clever things for memory allocation, whereas my array algebra
    //   code that commented out here allocates new arrays.
    ////////////////////////////////////////////////////

    //let diffs = arr1 - &arr2.insert_axis(Axis(2));
    //let weighted_diff = diffs * &weights.insert_axis(Axis(2));
    //let abs_diffs = weighted_diff.mapv(|elem| elem.abs());
    //abs_diffs.sum_axis(Axis(0)).sum_axis(Axis(0))

    let fwd_diff = ndarray::Zip::from(arr1.slice(s![..,..,0])).
        and(arr2).
        and(weights).
        fold(0.0, |acc, a, b, c| acc + (a-b).abs()*c);

    let rev_diff = ndarray::Zip::from(arr1.slice(s![..,..,1])).
        and(arr2).
        and(weights).
        fold(0.0, |acc, a, b, c| acc + (a-b).abs()*c);

    ndarray::array![fwd_diff, rev_diff]
}

pub fn weighted_manhattan_distance(
    arr1: &ndarray::ArrayView::<f64, Ix2>, 
    arr2: &ndarray::ArrayView::<f64, Ix2>,
    weights: &ndarray::ArrayView::<f64, Ix2>
) -> f64 {

    ////////////////////////////////////////////////////
    // leave this here for now. It yields identical results to the code below.
    // We could check to see which version is more performant.
    // I'm currently just guessing that the iter methods that are currently used are faster
    //   because they probably do clever things for memory allocation, whereas my array algebra
    //   code that commented out here allocates new arrays.
    ////////////////////////////////////////////////////
    //let weighted_diff = (arr1 - arr2) * weights;
    //let abs_diffs = weighted_diff.mapv(|elem| elem.abs());
    //abs_diffs.sum()

    ndarray::Zip::from(arr1).
        and(arr2).
        and(weights).
        fold(0.0, |acc, a, b, c| acc + (a-b).abs()*c)
}

/// Function to compute inverse-logit element-wise for an array
///
/// # Arguments
///
/// - `a` - value to apply inverse-logit to
/// - 'alpha` - an optional lower limit to constrain returned values to
pub fn inv_logit(a: f64, alpha: Option<f64>) -> f64 {
    let lower = alpha.unwrap_or(0.0);
    lower + (1.0 - lower) * a.exp() / (1.0 + a.exp())
}

pub fn unique_cats(arr: ndarray::ArrayView<i64, Ix1>) -> Vec<i64> {
    arr.iter().unique().cloned().collect_vec()
}

/// Sorts the columns of each row in a hits array
pub fn sort_hits(hits: &mut Array2<i64>) {
    // get min of each row
    let min = hits.map_axis(ndarray::Axis(1), |r| cmp::min(r[0], r[1]));
    // get max of each row
    let max = hits.map_axis(ndarray::Axis(1), |r| cmp::max(r[0], r[1]));
    
    // re-assign the columns to each row's min and max
    hits.column_mut(0).assign(&min);
    hits.column_mut(1).assign(&max);
}


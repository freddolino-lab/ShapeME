use std::error::Error;
use std::hash::Hash;
use std::cmp::Ordering;
use std::collections::{HashMap, BTreeMap};
use std::cmp;
use std::iter;
use std::fs;
use std::io::{BufReader, BufWriter};
use std::ops::Deref;
use statrs::function::gamma;
// allow random iteration over RecordsDB
use rand::thread_rng;
use rand::seq::SliceRandom;
// ndarray stuff
use ndarray::prelude::*;
use ndarray::Array;
use std::iter::FromIterator;
// ndarray_stats exposes ArrayBase to useful methods for descriptive stats like min.
use ndarray_stats::QuantileExt;
use ndarray_stats::CorrelationExt;
use itertools::Itertools;
use statrs::statistics::Statistics;
use ndarray_npy; // we've cloned the git repo for this crate for stability
use serde_pickle::{de, ser};
use serde_json;
use serde::{Serialize, Deserialize};
// parallelization utilities provided by rayon crate
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
// check the source code for OrderedFloat and consider implementing it in lib.rs instead of using the ordered_float crate. If ordered_float stops existing someday, we don't want our code to break.
use ordered_float::OrderedFloat;
use std::time;

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
    #[should_panic]
    fn test_len_mismatch() {
        let seqA = "ACTGTCA";
        let seqB = "AC";
        let result5 = seq_hamming_distance(&seqA, &seqB).unwrap();
    }

    #[test]
    fn test_no_key_error() {
        let nonsense = "anfrlas";
        let answer = "Key 'N' not found in lut {'A': 0, 'C': 1, 'G': 2, 'T': 3}";
        let result = letter_seq_to_one_hot(&nonsense).unwrap_err();
        assert_eq!(result, answer);
    }

    #[test]
    fn test_hamming_dist() {
        let seqA = "ACTGTCA";
        let seqB = "actgtca";
        let seqC = "aCtgTca";
        let seqD = "agtgTca";
        let seqE = "ggaccgt";
        
        let answer1: u64 = 0;
        let answer2: u64 = 0;
        let answer3: u64 = 1;
        let answer4: u64 = 7;

        let result1 = seq_hamming_distance(&seqA, &seqB).unwrap();
        let result2 = seq_hamming_distance(&seqA, &seqC).unwrap();
        let result3 = seq_hamming_distance(&seqA, &seqD).unwrap();
        let result4 = seq_hamming_distance(&seqA, &seqE).unwrap();

        assert_eq!(result1, answer1);
        assert_eq!(result2, answer2);
        assert_eq!(result3, answer3);
        assert_eq!(result4, answer4);
    }

    #[test]
    fn test_one_hot_to_letter() {
        let answer = "ACGTGCA";
        let arr = array![
            [1, 0, 0, 0, 0, 0, 1], // A
            [0, 1, 0, 0, 0, 1, 0], // C
            [0, 0, 1, 0, 1, 0, 0], // T
            [0, 0, 0, 1, 0, 0, 0], // G
        ];
        let result = one_hot_to_letter_seq(&arr.view()).unwrap();
        assert_eq!(result, answer);
    }

    #[test]
    fn test_letter_to_one_hot() {
        let letter_seq = "ACGTGCA";
        let answer = array![
            [1, 0, 0, 0, 0, 0, 1], // A
            [0, 1, 0, 0, 0, 1, 0], // C
            [0, 0, 1, 0, 1, 0, 0], // T
            [0, 0, 0, 1, 0, 0, 0], // G
        ];
        let result = letter_seq_to_one_hot(&letter_seq).unwrap();
        assert_eq!(result, answer);
        let letter_seq = "acgtgca";
        let result = letter_seq_to_one_hot(&letter_seq).unwrap();
        assert_eq!(result, answer);
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

    fn set_up_recdb(nrecs: usize) -> RecordsDB {
        let mut seq_vec: Vec<StrandedSequence> = Vec::new();
        for i in 0..nrecs {
            seq_vec.push(set_up_stranded_sequence(i as f64, 1));
        }
        let y_vals: Array<i64, Ix1> = Array::from_vec(
            (0..nrecs)
            .map(|x| x as i64)
            .collect()
        );
        RecordsDB::new(seq_vec, y_vals)
    }

    #[test]
    fn test_batch_iter(){
        let this_db = set_up_recdb(6);
        for (i,batch) in this_db.batch_iter(3).enumerate() {
            println!("-------------------------------");
            println!("batch {}: {:?}", i, batch);
            println!("-------------------------------");
            if i == 0 {
                assert_eq!(batch.len(), 3);
            }
            if i == 1 {
                assert_eq!(batch.len(), 3);
            }
            if i == 2{
                panic!();
            }
        }
        for (i,batch) in this_db.batch_iter(6).enumerate() {
            println!("-------------------------------");
            println!("batch {}: {:?}", i, batch);
            println!("-------------------------------");
            if i == 0 {
                assert_eq!(batch.len(), 6);
            }
            if i == 1{
                panic!();
            }
        }
        for (i,batch) in this_db.batch_iter(5).enumerate() {
            println!("-------------------------------");
            println!("batch {}: {:?}", i, batch);
            println!("-------------------------------");
            if i == 0 {
                assert_eq!(batch.len(), 5);
            }
            if i == 1 {
                assert_eq!(batch.len(), 1);
            }
            if i == 2 {
                panic!();
            }
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

        // a_stranded_seq matches seed exactly
        let a_stranded_seq = set_up_stranded_sequence(2.0, length);
        // another_stranded_seq is off of seed by 0.1 at every position
        let another_stranded_seq = set_up_stranded_sequence(2.1, length);
        let hits = a_stranded_seq.count_hits_in_seq(
            &seed.params.params,
            &wv,
            &threshold2,
            &max_count,
        );
        println!("{:?}", hits);
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
        let hits = db.get_hits(
            &test_seed.params.params,
            &seeds.weights.weights_norm.view(),
            &1.0,
            &10,
        );
        assert_eq!(hits[[6,0]], 10)
    }

    #[test]
    fn test_recordb_ami(){
        let max_count: i64 = 10;
        let db = setup_RecordsDB(30, 30);
        let seeds = db.make_seed_vec(15, 0.01);
        let test_seed = &seeds.seeds[100];
        let hits = db.get_hits(
            &test_seed.params.params,
            &seeds.weights.weights_norm.view(),
            &1.0,
            &max_count,
        );
        
        let hit_cats = info_theory::categorize_hits(&hits, &max_count);
        let hv = hit_cats.view();
        let vv = db.values.view();
        let contingency = info_theory::construct_contingency_matrix(hv, vv);
        let ami = info_theory::adjusted_mutual_information(contingency.view());
        let ami_answer = 0.04326620722450172; // calculated using sklear.metrics.adjusted_mutual_info_score
        assert!(AbsDiff::default().epsilon(1e-6).eq(&ami_answer, &ami));
    }

    #[test]
    fn test_fold_merge_update() {

        let mut motifs: Motifs = read_motifs("/corexfs/schroedj/src/DNAshape_motif_finder/src/rust_utils/test_data/test_motifs.json");
        // simulates args as they'll come from env::args in main.rs
        let args = [
            String::from("motifer"),
            String::from("/corexfs/schroedj/src/DNAshape_motif_finder/src/rust_utils/test_data/config.json"),
        ];
        let cfg = parse_config(&args).unwrap();
        let rec_db = RecordsDB::new_from_files(
            &cfg.shape_fname,
            &cfg.yvals_fname,
        );

        println!("{:?}", motifs);
        motifs.fold_merge_update(&rec_db, &1, &0.01);
        println!("{:?}", motifs);
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
        let fname = "/corexfs/schroedj/src/DNAshape_motif_finder/src/rust_utils/test_data/shapes.npy";
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
        let fname = "/corexfs/schroedj/src/DNAshape_motif_finder/src/rust_utils/test_data/y_vals.npy";
        let y_vals: Array1<i64> = ndarray_npy::read_npy(fname).unwrap();
        assert_eq!((2000), y_vals.dim());
        assert_eq!(391, y_vals.sum());

        // read in hits for first record, first window, calculated in python
        // These values will be used to test whether our rust hit counting
        // yields the same results as our python hit counting.
        let fname = "/corexfs/schroedj/src/DNAshape_motif_finder/src/rust_utils/test_data/hits.npy";
        let hits: Array2<i64> = ndarray_npy::read_npy(fname).unwrap();
        assert_eq!((2000,2), hits.dim());
        assert_eq!(1323, hits.sum());

        // read in some other parameters we'll need
        let fname = "/corexfs/schroedj/src/DNAshape_motif_finder/src/rust_utils/test_data/test_args.pkl";
        let file = fs::File::open(fname).unwrap();
        // open a buffered reader to open the pickle file
        let buf_reader = BufReader::new(file);
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
        let shape_fname = String::from("/corexfs/schroedj/src/DNAshape_motif_finder/src/rust_utils/test_data/shapes.npy");
        // read in y-vals
        let y_fname = String::from("/corexfs/schroedj/src/DNAshape_motif_finder/src/rust_utils/test_data/y_vals.npy");
        let rec_db = RecordsDB::new_from_files(
            &shape_fname,
            &y_fname,
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
        // read in shapes
        let shape_fname = String::from("/corexfs/schroedj/src/DNAshape_motif_finder/src/rust_utils/test_data/shapes.npy");
        // read in y-vals
        let y_fname = String::from("/corexfs/schroedj/src/DNAshape_motif_finder/src/rust_utils/test_data/y_vals.npy");
        let rec_db = RecordsDB::new_from_files(
            &shape_fname,
            &y_fname,
        );

        // read in some other parameters we'll need
        let fname = "/corexfs/schroedj/src/DNAshape_motif_finder/src/rust_utils/test_data/test_args.pkl";
        let file = fs::File::open(fname).unwrap();
        // open a buffered reader to open the pickle file
        let buf_reader = BufReader::new(file);
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
        // when I made the file "../test_data/distances.npy"
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
            String::from("/corexfs/schroedj/src/DNAshape_motif_finder/src/rust_utils/test_data/distances.npy")
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
        let fname = "/corexfs/schroedj/src/DNAshape_motif_finder/src/rust_utils/test_data/hits.npy";
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
            String::from("/corexfs/schroedj/src/DNAshape_motif_finder/src/rust_utils/test_data/config.json"),
        ];
        let cfg = parse_config(&args).unwrap();
        let rec_db = RecordsDB::new_from_files(
            &cfg.shape_fname,
            &cfg.yvals_fname,
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
    fn test_read_motifs () {
        let motifs: Motifs = read_motifs("/corexfs/schroedj/src/DNAshape_motif_finder/src/rust_utils/test_data/test_motifs.json");
        println!("{:?}", motifs.motifs[0].params.params);
        let motifs: Motifs = read_motifs("/corexfs/schroedj/src/DNAshape_motif_finder/src/rust_utils/test_data/test_motifs_err.json");
        println!("{:?}", motifs.motifs[0].params.params);
    }

    #[test]
    fn test_init_threshold () {
        // simulates args as they'll come from env::args in main.rs
        let args = [
            String::from("motifer"),
            String::from("/corexfs/schroedj/src/DNAshape_motif_finder/src/rust_utils/test_data/config_init_thresh.json"),
        ];

        let cfg = parse_config(&args).unwrap();
        let rec_db = RecordsDB::new_from_files(
            &cfg.shape_fname,
            &cfg.yvals_fname,
        );

        // create Seeds struct
        let seeds = rec_db.make_seed_vec(cfg.kmer, cfg.alpha);
        let thresh = set_initial_threshold(
            &seeds,
            &rec_db,
            &cfg.seed_sample_size,
            &cfg.records_per_seed,
            &cfg.windows_per_record,
            &cfg.kmer,
            &cfg.alpha,
            &cfg.thresh_sd_from_mean,
        );
        // assert that the python and rust estimates are within 0.03 of each other
        assert!(
            AbsDiff::default().epsilon(0.03).eq(&thresh, &cfg.threshold)
        );
        println!("Rust initial threshold: {}", thresh);
        println!("Python initial threshold: {}", cfg.threshold);
    }
}

/// Simple struct to hold command line arguments
#[derive(Serialize, Deserialize, Debug)]
pub struct Config {
    #[serde(default = "default_fname")]
    pub shape_fname: String,
    #[serde(default = "default_fname")]
    pub yvals_fname: String,
    #[serde(default = "default_fname")]
    pub eval_shape_fname: String,
    #[serde(default = "default_fname")]
    pub eval_yvals_fname: String,
    #[serde(default = "default_fname")]
    pub out_fname: String,
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    #[serde(default = "default_max_count")]
    pub max_count: i64,
    #[serde(default = "default_kmer")]
    pub kmer: usize,
    #[serde(default = "default_thresh")]
    pub threshold: f64,
    #[serde(default = "default_core_num")]
    pub cores: usize,
    #[serde(default = "default_seed_sample_size")]
    pub seed_sample_size: usize,
    #[serde(default = "default_records_per_seed")]
    pub records_per_seed: usize,
    #[serde(default = "default_windows_per_record")]
    pub windows_per_record: usize,
    #[serde(default = "default_thresh_sd_from_mean")]
    pub thresh_sd_from_mean: f64,
    #[serde(default = "default_thresh_lb")]
    pub thresh_lower_bound: f64,
    #[serde(default = "default_thresh_ub")]
    pub thresh_upper_bound: f64,
    #[serde(default = "default_shape_lb")]
    pub shape_lower_bound: f64,
    #[serde(default = "default_shape_ub")]
    pub shape_upper_bound: f64,
    #[serde(default = "default_weight_lb")]
    pub weight_lower_bound: f64,
    #[serde(default = "default_weight_ub")]
    pub weight_upper_bound: f64,
    #[serde(default = "default_temp")]
    pub temperature: f64,
    #[serde(default = "default_stepsize")]
    pub stepsize: f64,
    #[serde(default = "default_n_opt_iter")]
    pub n_opt_iter: usize,
    #[serde(default = "default_tadj")]
    pub t_adjust: f64,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_names")]
    pub names: Vec<String>,
    #[serde(default = "default_indices")]
    pub indices: Vec<usize>,
    #[serde(default = "default_centers")]
    pub centers: Vec<f64>,
    #[serde(default = "default_spreads")]
    pub spreads: Vec<f64>,
    #[serde(default = "default_fname")]
    pub motif_fname: String,
    #[serde(default = "default_fname")]
    pub logit_reg_fname: String,
    #[serde(default = "default_fname")]
    pub eval_rust_fname: String,
    #[serde(default = "default_max_batches")]
    pub max_batch_no_new: usize,
}

fn default_fname() -> String { String::from("default") }
fn default_names() -> Vec<String> { vec![String::from("default")] }
fn default_indices() -> Vec<usize> { vec![0] }
fn default_centers() -> Vec<f64> { vec![-1000.0] }
fn default_spreads() -> Vec<f64> { vec![-1000.0] }
fn default_seed_sample_size() -> usize { 250 }
fn default_records_per_seed() -> usize { 50 }
fn default_windows_per_record() -> usize { 1 }
fn default_thresh_sd_from_mean() -> f64 { 2.0 }
fn default_core_num() -> usize { 1 }
fn default_max_count() -> i64 { 1 }
// leave this threshold default value as is (&0.8711171869882366)!
// It's only used in our unit tests, and it
// makes them work!!
fn default_thresh() -> f64 { 0.8711171869882366 }
fn default_kmer() -> usize { 15 }
fn default_alpha() -> f64 { 0.01 }
fn default_batch_size() -> usize { 2000 }
fn default_tadj() -> f64 { 0.0005 }
fn default_n_opt_iter() -> usize { 12000 }
fn default_stepsize() -> f64 { 0.25 }
fn default_temp() -> f64 { 0.2 }
fn default_weight_ub() -> f64 { 4.0 }
fn default_weight_lb() -> f64 { -4.0 }
fn default_shape_ub() -> f64 { 4.0 }
fn default_shape_lb() -> f64 { -4.0 }
fn default_thresh_ub() -> f64 { 4.0 }
fn default_thresh_lb() -> f64 { 0.0 }
fn default_max_batches() -> usize { 10 }

impl Config {
    /// Returns a Config struct containing options contained in
    /// command line arguments
    ///
    /// # Arguments
    ///
    /// * `args` - an array of [String] structs. Comes from env::args in main.rs
    ///
    /// STILL NEEDS GOOD ERROR HANDLING, IMO
    pub fn new(args: &[String]) -> Result<Config, Box<dyn Error>> {
        let opts_fname = args[1].clone();

        // read in options we'll need
        let file = fs::File::open(opts_fname)?;
        // open a buffered reader to open the binary json file
        let buf_reader = BufReader::new(file);
        let cfg: Config = serde_json::from_reader(buf_reader)?;
        Ok(cfg)
    }

    pub fn write(&self, fname: &str) -> Result<(), Box<dyn Error>> {
        // set up writer
        let file = fs::File::create(fname).unwrap();
        // open a buffered writer to open the pickle file
        let mut buf_writer = BufWriter::new(file);
        let j = serde_json::to_writer_pretty(buf_writer, &self)?;
        Ok(())
    }
}

pub fn wrangle_params_for_optim(
        motif: &Motif,
        shape_lower_bound: &f64,
        shape_upper_bound: &f64,
        weights_lower_bound: &f64,
        weights_upper_bound: &f64,
        threshold_lower_bound: &f64,
        threshold_upper_bound: &f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut params = Vec::new();
    let mut lower_bound = Vec::new();
    let mut upper_bound = Vec::new();

    let shapes = motif.params.params.to_owned().into_raw_vec();
    let shape_len = shapes.len();
    params.extend(shapes);
    lower_bound.extend(vec![*shape_lower_bound; shape_len]);
    upper_bound.extend(vec![*shape_upper_bound; shape_len]);

    let weights = motif.weights.weights.to_owned().into_raw_vec();
    params.extend(weights);
    lower_bound.extend(vec![*weights_lower_bound; shape_len]);
    upper_bound.extend(vec![*weights_upper_bound; shape_len]);

    params.push(motif.threshold);
    lower_bound.push(*threshold_lower_bound);
    upper_bound.push(*threshold_upper_bound);

    (params, lower_bound, upper_bound)
}

pub fn opt_vec_to_motif(
        params_vec: &Vec<f64>,
        rec_db: &RecordsDB,
        alpha: &f64,
        max_count: &i64,
        kmer: &usize,
) -> Motif {

    // get the number of shape params and window size
    let shape_num = rec_db.seqs[0].params.raw_dim()[0];
    let record_num = rec_db.len();
    let slice_length = shape_num * kmer;
    // create [Sequence] instance with optimized shape values
    let opt_params = Sequence {
        params: ndarray::Array::from_shape_vec(
            (shape_num, *kmer),
            params_vec[0..slice_length].to_vec(),
        ).unwrap(),
    };
    // get optimized values for the weights field
    let these_weights = ndarray::Array::from_shape_vec(
        (shape_num, *kmer),
        params_vec[slice_length..slice_length*2].to_vec(),
    ).unwrap();
    let opt_threshold = params_vec[params_vec.len()-1];
    // finally, instantiate the Motif
    let mut motif = Motif::new(
        opt_params,
        opt_threshold,
        record_num,
    );
    // set the Motif's weights. constrain_normalize is called within set_weights
    motif.weights.set_weights(these_weights.view(), &alpha);
    // now set its hits and mi
    let weights = motif.weights.weights_norm.to_owned();
    motif.update_info(
        &rec_db,
        &weights.view(),
        &opt_threshold,
        max_count,
    );

    motif
}

/// Objective function for optimization.
///
/// # Arguments
///
/// * `params` - The parameters for a motif,
///      appended to each other and flattened to a Vec
/// * `kmer` - The window size for our Motif instances
/// * `rec_db` - A reference to our RecordsDB instance
/// * `max_count` - The max count for hits counting. A reference to i64.
/// * `alpha` - The lower bound on the normalized weights. Reference to f64.
///
/// # Returns
///
/// * `score` - negative adjusted mutual information
pub fn optim_objective(
        params: &Vec<f64>,
        rec_db: &RecordsDB,
        kmer: &usize,
        max_count: &i64,
        alpha: &f64,
) -> f64 {
    
    let shape_num = rec_db.seqs[0].params.raw_dim()[0];
    let length = kmer * shape_num;

    // view to slice of params containing shapes
    let shape_view = ArrayView::from_shape(
        (shape_num, *kmer),
        &params[0..length],
    ).unwrap();

    // view to slice of params containing shapes
    let weights_arr = ArrayView::from_shape(
        (shape_num, *kmer),
        &params[length..2 * length],
    ).unwrap().to_owned();

    // threshold is the final element in the params Vec
    let threshold = params[params.len() - 1];

    // create a SequenceView so that we can then create a MotifWeights instance
    let seq_view = SequenceView::new(shape_view);
    let mut motif_weights = MotifWeights::new(&seq_view);
    // set and normalize the weights
    motif_weights.set_weights(weights_arr.view(), &alpha);

    // get the hits
    let hits = rec_db.get_hits(
        &shape_view,
        &motif_weights.weights_norm.view(),
        &threshold,
        max_count,
    );
    // categorize the hits and calculate adjusted mutual information
    let hit_cats = info_theory::categorize_hits(&hits, &max_count);
    let contingency = info_theory::construct_contingency_matrix(
        hit_cats.view(),
        rec_db.values.view(),
    );

    -info_theory::adjusted_mutual_information(contingency.view())
}

/// Parses arguments passed at the command line and places them into a [Config]
pub fn parse_config(args: &[String]) -> Result<Config, Box<dyn Error>> {
    Ok(Config::new(args)?)
}

/// Randomly chooses pairs of seeds and returns the CMI for each pair
pub fn sample_cmi_vals(
        n: usize,
        seeds: &Seeds,
        rec_db: &RecordsDB,
        max_count: &i64,
) -> Vec<f64> {

    let mut cmi_vec = Vec::<f64>::new();
    let seed_vec = &seeds.seeds;
    let mut inds: Vec<usize> = (0..seed_vec.len()).collect();
    // grab random n indices
    inds.shuffle(&mut thread_rng());
    let x_inds = &inds[0..n].to_vec();
    // grab another random n indices
    inds.shuffle(&mut thread_rng());
    let y_inds = &inds[0..n].to_vec();

    x_inds.iter().zip(y_inds).for_each(|(x,y)| {
        let x_seed = &seed_vec[*x];
        let y_seed = &seed_vec[*y];
        let x_cats = info_theory::categorize_hits(&x_seed.hits, max_count);
        let y_cats = info_theory::categorize_hits(&y_seed.hits, max_count);

        let contingency = info_theory::construct_3d_contingency(
            x_cats.view(),
            rec_db.values.view(),
            y_cats.view(),
        );

        let cmi = info_theory::conditional_adjusted_mutual_information(
            contingency.view()
        );
        cmi_vec.push(cmi);
    });
    cmi_vec
}

pub fn gamma_objective(params: &Vec<f64>, data: &Vec<f64>) -> f64 {
    let ecdf = get_ecdf(data, 0.8);
    let fitted = get_fitted_gamma_cdf_vals(params, data);
    sum_squared_residuals(&ecdf, &fitted)
}

pub fn beta_logp_objective(params: &Vec<f64>, data: &Vec<f64>) -> f64 {
    // sort the data
    let mut sorted_vals = data.to_vec();
    sorted_vals.par_sort_unstable_by_key(|val| OrderedFloat(*val));

    // take the lowest `retain_lower_p` fraction of data to get ecdf
    let final_idx = (sorted_vals.len() as f64 * 0.8) as usize;

    let logp: f64 = sorted_vals[0..final_idx].iter()
        .map(|x| (beta_pdf(params, x) * 1.0/0.8).ln())
        .sum();
    -logp
}

fn beta_pdf(params: &Vec<f64>, x: &f64) -> f64 {
    let numer1 = x.powf(params[0]-1.0);
    let numer2 = (1.0 - x).powf(params[1]-1.0);
    let numer = numer1 * numer2;
    let denom = gamma::ln_gamma(params[0]).exp()
        * gamma::ln_gamma(params[1]).exp()
        / gamma::ln_gamma(params[0] + params[1]).exp();
    numer/denom
}

pub fn get_fitted_beta_pdf(params: &Vec<f64>, data: &Vec<f64>) -> Vec<f64> {
    data.iter()
        .map(|x| beta_pdf(params, x))
        .collect()
}

pub fn get_ecdf(data: &Vec<f64>, retain_lower_p: f64) -> Vec<f64> {

    // vector to populate with value of ecdf for each value in data
    let mut ecdf = Vec::<f64>::new();
    let n = data.len();

    // sort the data
    let mut sorted_vals = data.to_vec();
    sorted_vals.par_sort_unstable_by_key(|val| OrderedFloat(*val));

    // take the lowest `retain_lower_p` fraction of data to get ecdf
    let final_idx = (sorted_vals.len() as f64 * retain_lower_p) as usize;
    for (i,_) in sorted_vals[0..final_idx].iter().enumerate() {
        let p_leq = (i + 1) as f64 / n as f64;
        ecdf.push(p_leq);
    }
    ecdf
}

fn sum_squared_residuals(ecdf: &Vec<f64>, fitted: &Vec<f64>) -> f64 {
    ecdf.iter()
        .zip(fitted)
        .map(|(a,b)| (a - b).powf(2.0_f64))
        .sum()
}

pub fn get_fitted_gamma_cdf_vals(params: &Vec<f64>, data: &Vec<f64>) -> Vec<f64> {
    let min_val = data.to_vec().into_iter().reduce(f64::min).unwrap();
    data.iter()
        .map(|x| {
            // if the minimum cmi is less than zero, shift dist by -min_val+epsilon
            let mut x_i = f64::EPSILON;
            if min_val <= 0.0 {
                x_i = *x - min_val + f64::EPSILON;
            }
            let beta_x = params[1] * x_i;
            let numer = gamma::gamma_li(params[0], beta_x);
            let denom = gamma::ln_gamma(params[0]).exp();
            numer/denom
        })
        .collect()
}

/// Filters motifs based on conditional mutual information
pub fn filter_motifs<'a>(
        motifs: &'a mut Vec<Motif>,
        rec_db: &'a RecordsDB,
        //threshold: &'a f64,
        max_count: &'a i64,
) -> Motifs {

    // get number of parameters in model (shape number * seed length * 2)
    //  we multiply by two because we'll be optimizing shape AND weight values
    //  then add one for the threshold
    let rec_num = rec_db.len();
    let shape_num = motifs[0].params.params.raw_dim()[0];
    let win_len = motifs[0].params.params.raw_dim()[1];
    let delta_k = shape_num * win_len * 2 + 1;
    //let mut info_vals_in_model = Vec::<&f64>::new();

    // sort the Vec of Motifs in order of descending mutual information
    motifs.par_sort_unstable_by_key(|motif| OrderedFloat(-motif.mi));
    let mut top_motifs = Vec::new();

    // Make sure first seed passes AIC
    let log_lik = rec_num as f64 * motifs[0].mi;
    let aic = info_theory::calc_aic(delta_k, log_lik);
    if aic < 0.0 {
        let motif = motifs[0].to_motif();
        top_motifs.push(motif);
        //info_vals_in_model.push(aic);
    } else {
        return Motifs::new(top_motifs)
    }

    // loop through candidate motifs
    for cand_motif in motifs[1..motifs.len()].iter() {

        // if this motif doesn't pass AIC on its own, with delta_k params, skip it
        let log_lik = rec_num as f64 * cand_motif.mi;
        if info_theory::calc_aic(delta_k, log_lik) > 0.0 {
            continue
        }

        let cand_hits = &cand_motif.hits;
        let cand_cats = info_theory::categorize_hits(&cand_hits, max_count);
        let mut motif_pass = true;

        for good_motif in top_motifs.iter() {

            // check the conditional mutual information for this seed with
            //   each of the chosen seeds
            let good_motif_hits = &good_motif.hits;
            let good_cats = info_theory::categorize_hits(&good_motif_hits, max_count);

            let contingency = info_theory::construct_3d_contingency(
                cand_cats.view(),
                rec_db.values.view(),
                good_cats.view(),
            );
            let cmi = info_theory::conditional_adjusted_mutual_information(
                contingency.view()
            );

            let param_num = delta_k * (top_motifs.len() + 1);
            //let proposed_info = info_vals_in_model.iter().sum() + cmi;
            let log_lik = rec_num as f64 * cmi;
            let this_aic = info_theory::calc_aic(param_num, log_lik);

            // if candidate seed doesn't improve model as added to each of the
            //   chosen seeds, skip it
            if this_aic > 0.0 {
                motif_pass = false;
                break
            }
        }
        if motif_pass {
            top_motifs.push(cand_motif.to_motif());
        }
    }
    Motifs::new(top_motifs)
}

/// Filters seeds based on conditional mutual information
pub fn filter_seeds<'a>(
        seeds: &mut Seeds<'a>,
        rec_db: &'a RecordsDB,
        threshold: &f64,
        max_count: &i64,
) -> Motifs {

    // get number of parameters in model (shape number * seed length * 2)
    //  we multiply by two because we'll be optimizing shape AND weight values
    //  then add one for the threshold
    let rec_num = rec_db.len();
    let delta_k = seeds.seeds[0].params.params.raw_dim()[1]
        * seeds.seeds[0].params.params.raw_dim()[0]
        * 2 + 1;
    //let mut info_vals_in_model = Vec::<&f64>::new();

    seeds.sort_seeds();

    let mut top_motifs = Vec::<Motif>::new();

    // Make sure first seed passes AIC
    let log_lik = rec_num as f64 * seeds.seeds[0].mi;
    let aic = info_theory::calc_aic(delta_k, log_lik);
    if aic < 0.0 {
        let motif = seeds.seeds[0].to_motif(threshold);
        top_motifs.push(motif);
        //info_vals_in_model.push(&seeds.seeds[0].mi);
    } else {
        return Motifs::new(top_motifs)
    }

    // loop through candidate seeds
    for (i,cand_seed) in seeds.seeds[1..seeds.seeds.len()].iter().enumerate() {

        //let now = time::Instant::now();
        // if this seed doesn't pass AIC on its own delta_k params, skip it
        let log_lik = rec_num as f64 * cand_seed.mi;
        if info_theory::calc_aic(delta_k, log_lik) > 0.0 {
            continue
        }

        let cand_hits = &cand_seed.hits;
        let cand_cats = info_theory::categorize_hits(&cand_hits, max_count);
        let mut seed_pass = true;

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

            // adjust number of parameters according to how many motifs will
            // be in the model if this one is added.
            let param_num = delta_k * (top_motifs.len() + 1);
            // add cmi to sum of current info in model to get proposed info
            //let proposed_info = info_vals_in_model.iter().sum() + cmi;
            // calculate log_likelihood-like factor
            let log_lik = rec_num as f64 * cmi;
            let this_aic = info_theory::calc_aic(param_num, log_lik);

            // if candidate seed doesn't improve model
            if this_aic > 0.0 {
                seed_pass = false;
                break
            }
        }
        if seed_pass {
            //println!("Seed {} passed filter. Appending to motif vec.", i+2);
            let motif = cand_seed.to_motif(threshold);
            top_motifs.push(motif);
            //info_vals_in_model.push(&cmi);
        } else {
            //println!("Seed {} did not pass filter.", i+2);
        }
        //let duration = now.elapsed().as_secs_f64() / 60.0;
        //println!("Evaluating whether seed {} took {} minutes.", i+2, duration);
    }
    Motifs::new(top_motifs)
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
#[derive(Debug, Clone, Serialize)]
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
    pub hits: ndarray::Array2::<i64>,
    pub mi: f64,
    pub dists: ndarray::Array2::<f64>,
    positions: Vec<HashMap<String,Vec<usize>>>,
    zscore: Option<f64>,
    robustness: Option<(u8,u8)>,
}

#[derive(Debug, Deserialize)]
struct ShapeReader {
    params: ArrayDeser<f64>,
}

#[derive(Debug, Deserialize)]
struct ArrayDeser<T> {
    v: usize,
    dim: (usize,usize),
    data: Vec<T>,
}

impl<T: std::clone::Clone> ArrayDeser<T> {
    fn to_array(&self) -> Array2<T> {
        Array::from_shape_vec(self.dim, self.data.to_vec()).unwrap()
    }
}

#[derive(Debug, Deserialize)]
struct WeightsReader {
    weights: ArrayDeser<f64>,
    weights_norm: ArrayDeser<f64>,
}

/// A struct for reading in Motifs data from a json file.
#[derive(Debug, Deserialize)]
pub struct MotifReader {
    params: ShapeReader,
    weights: WeightsReader,
    threshold: f64,
    hits: Option<ArrayDeser<i64>>,
    mi: Option<f64>,
    dists: Option<ArrayDeser<f64>>,
    positions: Option<Vec<HashMap<String, Vec<usize>>>>,
    zscore: Option<f64>,
    robustness: Option<(u8,u8)>,
}

/// Reads a vector of Motif structs from a pickle file
pub fn read_motifs(motif_fname: &str) -> Motifs {
    // read in options we'll need
    let file = fs::File::open(motif_fname).unwrap();
    // open a buffered reader to open the binary json file
    let buf_reader = BufReader::new(file);
    // create a hashmap from the pickle file's contents
    let motif_reader: Vec<MotifReader> = serde_json::from_reader(
        buf_reader,
    ).unwrap();
    Motifs::from_reader(motif_reader)
}

/// Represents a vector of Motif structs. It's basically just for convenience.
///
/// # Fields
///
/// * `motifs` - A Vec of Motif structs
#[derive(Debug, Serialize, Deserialize)]
pub struct Motifs {
    pub motifs: Vec<Motif>,
}

impl Motifs {
    pub fn empty() -> Motifs {
        Motifs{motifs: Vec::<Motif>::new()}
    }

    fn new(motifs: Vec<Motif>) -> Motifs {
        Motifs{motifs}
    }

    fn from_reader(reader: Vec<MotifReader>) -> Motifs {
        let mut motifs = Motifs::empty();
        for motif_reader in reader.iter() {
            let dim = motif_reader.params.params.dim;
            let params = Sequence{ params: motif_reader.params.params.to_array() };
            let mut weights = MotifWeights::new_bysize(dim.0, dim.1);
            weights.set_all_weights(
                &motif_reader.weights.weights.to_array(),
                &motif_reader.weights.weights_norm.to_array(),
            );
            let threshold = motif_reader.threshold;
            let hits = 
                if let Some(hits) = &motif_reader.hits {
                    hits.to_array()
                } else {
                // default hits is zero
                    Array2::zeros((dim.0,2))
                };
            let mi = 
                if let Some(mi) = motif_reader.mi {
                    mi
                } else {
                // default value for mi is 0.0
                    0.0_f64
                };
            let dists = 
                if let Some(dists) = &motif_reader.dists {
                    dists.to_array()
                // default value for dist is max for f64
                } else {
                    Array::from_elem((dim.0,2), f64::INFINITY)
                };
            let positions = 
                if let Some(positions) = &motif_reader.positions {
                    positions.to_vec()
                } else {
                    vec![HashMap::from([
                        (String::from("placeholder"), vec![usize::MAX])
                    ])]
                };
            let zscore = motif_reader.zscore;
            let robustness = motif_reader.robustness;
            let motif = Motif {
                params,
                weights,
                threshold,
                hits,
                mi,
                dists,
                positions,
                zscore,
                robustness,
            };
            motifs.motifs.push(motif);
        }
        motifs
    }

    pub fn append(&mut self, mut other: Motifs) {
        self.motifs.append(&mut other.motifs)
    }

    pub fn len(&self) -> usize {self.motifs.len()}

    pub fn supplement_robustness(&mut self, rec_db: &RecordsDB, max_count: &i64) {
        for (i,motif) in self.motifs.iter_mut().enumerate() {
            println!("Calculating robustness and z-score for motif {}", i);
            motif.update_robustness(rec_db, max_count);
            motif.update_zscore(rec_db, max_count);
        }
    }

    pub fn post_optim_update(&mut self, rec_db: &RecordsDB, max_count: &i64) {
        for (i,motif) in self.motifs.iter_mut().enumerate() {
            println!("Calculating final distances, mutual information, robustness, and z-score for motif {}", i);
            motif.update_min_dists(rec_db);
            motif.update_hit_positions(rec_db, max_count);
            motif.update_robustness(rec_db, max_count);
            motif.update_zscore(rec_db, max_count);
        }
    }

    pub fn fold_merge_update(
            &mut self,
            rec_db: &RecordsDB,
            max_count: &i64,
            alpha: &f64,
    ) {
        for (i,motif) in self.motifs.iter_mut().enumerate() {
            motif.normalize_weights(alpha);
            let hits = rec_db.get_hits(
                &motif.params.params.view(),
                &motif.weights.weights_norm.view(),
                &motif.threshold,
                max_count,
            );
            //println!("hits shape: {:?}", &hits.shape());
            motif.hits = hits;
            motif.update_min_dists(rec_db);
            motif.update_hit_positions(rec_db, max_count);
            motif.update_robustness(rec_db, max_count);
            motif.update_zscore(rec_db, max_count);
            motif.update_mi(rec_db, max_count);
        }
    }

    fn gather_dist_arr(&self) -> Array2<f64> {
        let mut all_dists = Array2::<f64>::zeros(
            (self.len(), self.motifs[0].dists.len())
        );
        for (i, mut row) in all_dists.axis_iter_mut(Axis(0)).enumerate() {
            let dists_i = Array::from_iter(self.motifs[i].dists.iter().cloned());
            row.assign(&dists_i);
        }
        all_dists
    }

    pub fn get_motif_correlations(&self) -> Array2<f64> {
        let all_dists = self.gather_dist_arr();
        let mut corr = all_dists.pearson_correlation().unwrap();
        // make corr upper triangular and set diag to 0.0
        for i in 0..self.len() {
            corr.row_mut(i).slice_mut(s![0..i+1]).fill(0.0);
        }
        corr
    }

    /// Writes a vector of Motif structs as a pickle file
    pub fn pickle_motifs(&self, pickle_fname: &str) {
        // set up writer
        let file = fs::File::create(pickle_fname).unwrap();
        // open a buffered writer to open the pickle file
        let mut buf_writer = BufWriter::new(file);
        // write to the writer
        let res = ser::to_writer(
            &mut buf_writer, 
            &self.motifs, 
            ser::SerOptions::new(),
        );
    }

    /// Writes a vector of Motif structs as a json file
    pub fn json_motifs(&self, fname: &str) {
        // set up writer
        let file = fs::File::create(fname).unwrap();
        // open a buffered writer to open the pickle file
        let mut buf_writer = BufWriter::new(file);
        // write to the writer
        serde_json::to_writer(
            &mut buf_writer, 
            &self.motifs, 
        );
    }

    fn sort_motifs(&mut self) {
        self.motifs.par_sort_unstable_by_key(|motif| OrderedFloat(-motif.mi))
    }

    /// Filters motifs based on conditional mutual information
    pub fn filter_motifs<'a>(
            &mut self,
            rec_db: &'a RecordsDB,
            max_count: &'a i64,
    ) -> Motifs {

        if self.len() == 0 {
            return Motifs::empty();
        }
        // get number of parameters in model (shape number * seed length * 2)
        //  we multiply by two because we'll be optimizing shape AND weight values
        //  then add one for the threshold
        let rec_num = rec_db.len();
        println!("rec_num: {}", &rec_num);
        let shape_num = self.motifs[0].params.params.raw_dim()[0];
        println!("shape_num: {}", &shape_num);
        let win_len = self.motifs[0].params.params.raw_dim()[1];
        println!("win_len: {}", &win_len);
        let delta_k = shape_num * win_len * 2 + 1;
        println!("delta_k: {}", &delta_k);
        //let mut info_vals_in_model = Vec::<&f64>::new();

        // sort the Vec of Motifs in order of descending mutual information
        self.sort_motifs();

        for (k,motif) in self.motifs.iter().enumerate() {
            println!("Motif {k} AMI is {}", motif.mi);
        }

        let mut top_motifs = Vec::new();

        // Make sure first seed passes AIC
        let log_lik = rec_num as f64 * self.motifs[0].mi;
        let aic = info_theory::calc_aic(delta_k, log_lik);

        println!("AIC: {}", &aic);

        if aic < 0.0 {
            let motif = self.motifs[0].to_motif();
            top_motifs.push(motif);
            //info_vals_in_model.push(aic);
        } else {
            return Motifs::new(top_motifs)
        }

        // loop through candidate motifs
        for cand_motif in self.motifs[1..self.motifs.len()].iter() {

            // if this motif doesn't pass AIC on its own, with delta_k params, skip it
            let log_lik = rec_num as f64 * cand_motif.mi;
            if info_theory::calc_aic(delta_k, log_lik) > 0.0 {
                continue
            }

            let cand_hits = &cand_motif.hits;
            let cand_cats = info_theory::categorize_hits(&cand_hits, max_count);
            let mut motif_pass = true;

            for good_motif in top_motifs.iter() {

                // check the conditional mutual information for this seed with
                //   each of the chosen seeds
                let good_motif_hits = &good_motif.hits;
                let good_cats = info_theory::categorize_hits(&good_motif_hits, max_count);

                let contingency = info_theory::construct_3d_contingency(
                    cand_cats.view(),
                    rec_db.values.view(),
                    good_cats.view(),
                );
                let cmi = info_theory::conditional_adjusted_mutual_information(
                    contingency.view()
                );

                let param_num = delta_k * (top_motifs.len() + 1);
                //let proposed_info = info_vals_in_model.iter().sum() + cmi;
                let log_lik = rec_num as f64 * cmi;
                let this_aic = info_theory::calc_aic(param_num, log_lik);

                // if candidate seed doesn't improve model as added to each of the
                //   chosen seeds, skip it
                if this_aic > 0.0 {
                    motif_pass = false;
                    break
                }
            }
            if motif_pass {
                top_motifs.push(cand_motif.to_motif());
                //info_vals_in_model.push(&cmi);
            }
        }
        Motifs::new(top_motifs)
    }
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

/// Contains information about a given seed, including its sequence, hits array,
/// and adjusted mutual information. This struct does not own it shapes array.
/// Rather, we save a lot of memory by using a view to the relevant data
/// in the original RecordsDB struct.
///
/// # Fields
///
/// * `params` - An immutable SequenceView pointing to the
///     shape values for this seed
/// * `hits` - 2D array of shape (R,2), where R is the number of records
///     in the input data.
/// * `mi` - adjusted mutual information between this seeds' `hits` and
///     the original RecordsDB stuct's `values`. 
#[derive(Debug, Serialize)]
pub struct Seed<'a> {
    params: SequenceView<'a>,
    hits: ndarray::Array2::<i64>,
    mi: f64,
}

/// A container to hold all our seeds and one set of initial shared weights.
#[derive(Debug, Serialize)]
pub struct Seeds<'a> {
    pub seeds: Vec<Seed<'a>>,
    pub weights: MotifWeights
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
    pub values: ndarray::Array1::<i64>,
    inds: Vec<usize>
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
/// * `db` - A reference to the [RecordsDB]
/// * `sample_size` - Number of indices to take from the beginning of `indices`
/// * `indices` - a Vec of shuffled indices
#[derive(Debug)]
pub struct PermutedRecordsDBIter<'a> {
    loc: usize,
    db: &'a RecordsDB,
    sample_size: usize,
    indices: Vec<usize>,
}

/// Allows for iteration over batches of a records database
///
/// # Fields
///
/// * `loc` - Current batch index in the batched database
/// * `value` - A reference to the [RecordsDB]
/// * `batch_size` - Size of each batch
/// * `start_idx` - Starting index for a given batch
#[derive(Debug)]
pub struct BatchedRecordsDBIter<'a> {
    loc: usize,
    db: &'a RecordsDB,
    batch_size: usize,
    //final_batch_idx: usize,
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

    fn empty() -> StrandedSequence {
        let params = ndarray::Array3::zeros((1,1,1));
        StrandedSequence{ params }
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
    pub fn count_hits_in_seq(
            &self,
            query: &ndarray::ArrayView<f64,Ix2>,
            weights: &ndarray::ArrayView<f64, Ix2>,
            threshold: &f64,
            max_count: &i64,
    ) -> Array<i64, Ix1> {
    
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
    
    /// For a single Motif, get the positions at which we have hits
    ///
    /// # Arguments
    ///
    /// * `query` - an array which will be compared to each window in self.
    /// * `weights` - an array of weights to be applied for the distance calc
    /// * `threshold` - distance between the query and seq below which a hit is called
    /// * `max_count` - maximum number of times a hit will be counted on each strand
    pub fn get_hit_positions(
            &self,
            query: &ndarray::ArrayView<f64,Ix2>,
            weights: &ndarray::ArrayView<f64, Ix2>,
            threshold: &f64,
            max_count: &i64,
    ) -> HashMap<String, Vec<usize>> {
    
        // set maxed to false for each strand
        let mut f_maxed = false;
        let mut r_maxed = false;
        let mut positions = HashMap::from([
            (String::from("fwd"), Vec::new()),
            (String::from("rev"), Vec::new()),
        ]);
        let mut hits = Array1::<i64>::zeros(2);
    
        // iterate through windows of seq
        for (i,window) in self.window_iter(0, self.seq_len()+1, query.raw_dim()[1]).enumerate() {
    
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
                positions.get_mut("fwd").unwrap().push(i);
            } 

            if (dist[1] < *threshold) & (!r_maxed) {
                hits[1] += 1;
                if hits[1] == *max_count {
                    r_maxed = true;
                }
                positions.get_mut("rev").unwrap().push(i);
            } 
    
        }
        // return the positions
        positions
    }


    /// For a single Motif, get the minimum distance to each strand
    /// on the StrandedSequence.
    ///
    /// # Arguments
    ///
    /// * `query` - an array which will be compared to each window in self.
    /// * `weights` - an array of weights to be applied for the distance calc
    pub fn get_min_dists_in_seq(
            &self,
            query: &ndarray::ArrayView<f64,Ix2>,
            weights: &ndarray::ArrayView<f64, Ix2>,
    ) -> Array<f64, Ix1> {
    
        // initialize minimum distances at infinity
        let mut min_plus_dist = f64::INFINITY;
        let mut min_minus_dist = f64::INFINITY;
    
        // iterate through windows of seq
        for window in self.window_iter(0, self.seq_len()+1, query.raw_dim()[1]) {
    
            // get the distances.
            let dist = stranded_weighted_manhattan_distance(
                &window.params,
                query,
                weights,
            );

            if dist[0] < min_plus_dist {
                min_plus_dist = dist[0];
            }

            if dist[1] < min_minus_dist {
                min_minus_dist = dist[1];
            }
    
        }
        // return the minimum distances
        array![min_plus_dist, min_minus_dist]
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
        let r_maxed = true;
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
        // add the window width, self.size, to the start to get the end position
        //println!("This start: {}\nThis end: {}", this_start, this_end);
        // if we've reached the sample size, return None, exiting the iterator
        if self.start == self.sample_size {
            None
        } else {
            let this_start = self.indices[self.start];
            let this_end = this_start + self.size;
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
        let hits = ndarray::Array2::zeros((record_num, 2));
        let dists = Array::from_elem((record_num,2), f64::INFINITY);
        let positions = vec![HashMap::from([(String::from("placeholder"), vec![usize::MAX])])];
        let mi = 0.0;
        let zscore = None;
        let robustness = None;
        Motif{params, weights, threshold, hits, mi, dists, positions, zscore, robustness}
    }

    /// Returns a copy of Motif
    fn to_motif(&self) -> Motif {
        let mut weights = MotifWeights::new(&self.params.view());
        weights.set_all_weights(
            &self.weights.weights.to_owned(),
            &self.weights.weights_norm.to_owned(),
        );
        let hits = self.hits.to_owned();
        let dists = self.dists.to_owned();
        let positions = self.positions.to_owned();
        let arr = self.params.params.to_owned();
        let params = Sequence{ params: arr };
        let mi = self.mi;
        let threshold = self.threshold;
        let zscore = self.zscore;
        let robustness = self.robustness;
        Motif{params, weights, threshold, hits, mi, dists, positions, zscore, robustness}
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

    /// Similar to FIRE robustness score.
    fn update_robustness(&mut self, db: &RecordsDB, max_count: &i64) {
        let hit_cats = info_theory::categorize_hits(&self.hits, &max_count);
        let hv = hit_cats.view();
        let vv = db.values.view();
        let (num_passed,num_jacks) = info_theory::info_robustness(hv, vv);
        self.robustness = Some((num_passed, num_jacks));
    }

    /// Similar to FIRE z-score. Calculates z-score for
    /// jacknife replicates and places number that pass into Motif
    fn update_zscore(&mut self, db: &RecordsDB, max_count: &i64) {
        let hit_cats = info_theory::categorize_hits(&self.hits, &max_count);
        let hv = hit_cats.view();
        let vv = db.values.view();
        let (zscore,_) = info_theory::info_zscore(hv, vv);
        self.zscore = Some(zscore);
    }

    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    // NOTE: these update_* methods are duplicated in the Seed implementations!!
    // I was hoping we could write a trait to implement these methods within
    // the trait, and implement the trait for each of Motif and Seed, but
    // it looks like it isn't possible to have a trait access fields in
    // its methods. The compiler doesn't look ahead to see what structs
    // a trait is implemented for, so it doesn't "see" the fields.
    //////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    fn update_hits(&mut self, db: &RecordsDB,
                       weights: &ndarray::ArrayView<f64, Ix2>,
                       threshold: &f64,
                       max_count: &i64) {
        self.hits = db.get_hits(
            &self.params.params.view(),
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

    pub fn update_min_dists(
        &mut self,
        db: &RecordsDB,
    ) {
        self.dists = db.get_min_dists(
            &self.params.params.view(),
            &self.weights.weights_norm.view(),
        )
    }

    pub fn update_hit_positions(
        &mut self,
        db: &RecordsDB,
        max_count: &i64
    ) {
        self.positions = db.get_hit_positions(
            &self.params.params.view(),
            &self.weights.weights_norm.view(),
            &self.threshold,
            max_count,
        );
        self.hits = db.get_hits(
            &self.params.params.view(),
            &self.weights.weights_norm.view(),
            &self.threshold,
            max_count,
        )
    }
}

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
// NOTE: some of these impls are used in Motif as well. see not in the
// Motif implementations
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
impl<'a> Seed<'a> {

    pub fn new(params: SequenceView<'a>,
               record_num: usize) -> Seed<'a> {
        let hits = ndarray::Array2::zeros((record_num, 2));
        let mi = 0.0;
        Seed{params, hits, mi}
    }

    pub fn set_hits(&mut self, hits: ndarray::ArrayView<i64, Ix2>) {
        self.hits = hits.to_owned()
    }
    
    pub fn set_mi(&mut self, mi: f64) {
        self.mi = mi
    }

    fn update_hits(&mut self, db: &RecordsDB,
                       weights: &ndarray::ArrayView<f64, Ix2>,
                       threshold: &f64,
                       max_count: &i64) {
        //println!("Updating some hits");
        self.hits = db.get_hits(
            &self.params.params,
            weights,
            threshold,
            max_count,
        )
    }

    fn update_mi(&mut self, db: &RecordsDB, max_count: &i64) {
        //println!("Updating some mis");
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
        //println!("Updating some info");
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
        let hits = self.hits.to_owned();
        let dists = Array::from_elem(hits.raw_dim(), f64::INFINITY);
        let positions = vec![HashMap::from([(String::from("placeholder"), vec![usize::MAX])])];
        let arr = self.params.params.to_owned();
        let params = Sequence{ params: arr };
        let mi = self.mi;
        let zscore = None;
        let robustness = None;
        Motif {
            params,
            weights,
            threshold: *threshold,
            hits,
            mi,
            dists,
            positions,
            zscore,
            robustness,
        }
    }

}

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

    pub fn set_weights(
        &mut self,
        new_weights: ndarray::ArrayView<f64, Ix2>,
        alpha: &f64,
    ) {
        self.weights = new_weights.to_owned();
        self.constrain_normalize(alpha);
    }

    pub fn set_all_weights(
        &mut self,
        new_weights: &ndarray::Array<f64, Ix2>,
        new_weights_norm: &ndarray::Array<f64, Ix2>,
    ) {
        self.weights = new_weights.to_owned();
        self.weights_norm = new_weights_norm.to_owned();
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
            .and(&self.weights)
            .for_each(|a, b| *a = f64::exp(*b)/total);
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
            //println!("Done updating a seed\n=====================");
        });
        //println!("Done updating all seeds");

    }

    /// Sorts seeds by mi
    pub fn sort_seeds(&mut self) {
        self.seeds.par_sort_unstable_by_key(|seed| OrderedFloat(-seed.mi));
    }

    /// Writes a vector of Seed structs as a pickle file
    pub fn pickle_seeds(&self, pickle_fname: &str) {
        // set up writer
        let file = fs::File::create(pickle_fname).unwrap();
        // open a buffered writer to open the pickle file
        let mut buf_writer = BufWriter::new(file);
        // write to the writer
        let _res = ser::to_writer(
            &mut buf_writer, 
            &self.seeds, 
            ser::SerOptions::new(),
        );
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
        let inds = (0..seqs.len()).collect();
        RecordsDB{seqs, values, inds}
    }

    /// randomly shuffles records in self, returns the permuted order so that
    /// the random permutation can be un-done later
    pub fn permute_records(&mut self) {
        // get the vec of random indices
        let rand_inds: Vec<usize> = self.random_inds();
        let mut permuted_seqs: Vec<StrandedSequence> = Vec::with_capacity(self.len());
        let mut permuted_vals: ndarray::Array1::<i64> = Array1::zeros(self.len());
        for (i,rand_ind) in rand_inds.iter().enumerate() {
            let owned_params = self.seqs[*rand_ind].params.to_owned();
            permuted_seqs.push(StrandedSequence{ params: owned_params });
            permuted_vals[i] = self.values[*rand_ind];
        }
        self.seqs = permuted_seqs;
        self.values = permuted_vals;
        self.inds = rand_inds;
    }

    pub fn undo_record_permutation(&mut self) {
        let mut orig_seqs: Vec<StrandedSequence> = Vec::with_capacity(self.len());
        for _ in 0..self.len() {
            orig_seqs.push(StrandedSequence::empty());
        }
        let mut orig_vals: ndarray::Array1::<i64> = Array1::zeros(self.len());
        let mut used_indices: Vec<usize> = Vec::with_capacity(self.len());
        for (i, ind) in self.inds.iter().enumerate() {
            let params = self.seqs[i].params.to_owned();
            orig_seqs[*ind] = StrandedSequence::new(params);
            orig_vals[*ind] = self.values[i];
        }
        self.seqs = orig_seqs;
        self.values = orig_vals;
        self.inds = (0..self.len()).collect();
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
    pub fn new_from_files(shape_file: &str, y_val_file: &str) -> RecordsDB {
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

    fn random_inds(&self) -> Vec<usize> {
        // create vector of indices
        let mut inds: Vec<usize> = (0..self.seqs.len()).collect();
        // randomly shuffle the indices
        inds.shuffle(&mut thread_rng());
        inds
    }

    /// Permute the indices of the database, then iterate over the permuted indices
    pub fn random_iter(&self, sample_size: usize) -> PermutedRecordsDBIter {
        // create vector of indices
        let inds = self.random_inds();
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
    
    /// Iterate over batches of the RecordsDB
    pub fn batch_iter(&self, batch_size: usize) -> BatchedRecordsDBIter {
        BatchedRecordsDBIter{
            loc: 0,
            db: &self,
            batch_size: batch_size,
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
        max_count: &i64
    ) -> Array<i64, Ix2> {

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

        sort_hits(&mut hits);

        hits
    }
    
    /// Iterate over each record in the database and record the indices where
    /// `query` matches each record.
    ///
    /// # Arguments
    ///
    /// * `query` - A 2D arrayview, typically coming from a Seed's or Motif's params
    /// * `weights` - A 2D arrayview, typically coming from a Motif or a Seeds struct
    /// * `threshold` - The threshold distance below which a hit is called
    /// * `max_count` - The maximum number of hits to call for each strand
    pub fn get_hit_positions(
        &self,
        query: &ndarray::ArrayView<f64, Ix2>,
        weights: &ndarray::ArrayView<f64, Ix2>,
        threshold: &f64,
        max_count: &i64
    ) -> Vec<HashMap<String, Vec<usize>>> {

        let mut positions = Vec::new();
        self.seqs.iter().for_each(|seq| {
            let these_pos = seq.get_hit_positions(
                query,
                weights,
                threshold,
                max_count,
            );
            positions.push(these_pos);
        });

        positions
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
    pub fn get_min_dists(
        &self,
        query: &ndarray::ArrayView<f64, Ix2>,
        weights: &ndarray::ArrayView<f64, Ix2>,
    ) -> Array<f64, Ix2> {

        let mut dists = ndarray::Array2::zeros((self.len(), 2));
        dists.axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(&self.seqs)
            .for_each(|(mut row, seq)| {
                let this_dist = seq.get_min_dists_in_seq(
                    query,
                    weights,
                );
                row.assign(&this_dist);
            });

        dists
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

/// Enables batched iteration over a RecordsDB. Returns a [RecordsDB] as 
/// each item.
impl<'a> Iterator for BatchedRecordsDBIter<'a> {
    type Item = RecordsDB;

    fn next(&mut self) -> Option<Self::Item> {

        let mut end_idx = 0;

        if self.loc >= self.db.len() {
            None
        } else {

            end_idx = self.loc + self.batch_size;
            if end_idx >= self.db.len() {
                end_idx = self.db.len();
            }

            //let out_seqs = self.db.seqs[self.loc..end_idx].to_vec();
            let out_seqs = self.db.seqs.iter()
                .enumerate()
                .filter(|(i,_)| (i >= &self.loc) & (i < &end_idx))
                .map(|(_,seq)| StrandedSequence::new(seq.params.to_owned()))
                .collect();
                
            let out_vals = self.db.values.slice(s![self.loc..end_idx]).to_owned();

            self.loc += self.batch_size;
            Some(RecordsDB::new(out_seqs, out_vals))
        }
    }
}
/// Calculate distances for randomly chosen seeds and randomly selected
/// RecordsDBEntry structs
///
/// # Arguments
pub fn set_initial_threshold(
        seeds: &Seeds,
        rec_db: &RecordsDB,
        seed_sample_size: &usize,
        records_per_seed: &usize,
        windows_per_record: &usize,
        kmer: &usize,
        alpha: &f64,
        thresh_sd_from_mean: &f64,
) -> f64 {

    let seed_vec = &seeds.seeds;
    let mut inds: Vec<usize> = (0..seed_vec.len()).collect();
    inds.shuffle(&mut thread_rng());
    let keeper_inds = &inds[0..*seed_sample_size];

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
            for entry in rec_db.random_iter(*records_per_seed) {
                let seq = entry.seq;
                for window in seq.random_window_iter(0, seq.seq_len()+1, *kmer, *windows_per_record) {
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

/// Recodes a one-hot encoded array of shape (4,L), where L is the length
/// of seq to a letter sequence, i.e., a sequence of A, C, T, and Gs,
pub fn one_hot_to_letter_seq(arr: &ndarray::ArrayView<u64, Ix2>) -> Result<String, String> {
    let lut: BTreeMap<usize,char> = BTreeMap::from([
        (0, 'A'),
        (1, 'C'),
        (2, 'G'),
        (3, 'T'),
    ]);

    let categories = info_theory::one_hot_to_categorical(arr);
    let mut out_seq = String::new();
    for category in categories.iter() {
        let cat = *category as usize;
        let letter = lut.get(&cat);
        if letter.is_none() {
            return Err(format!(
                "Key '{}' not found in lut {:?}",
                &cat,
                &lut,
            ));
        };
        out_seq.push(*letter.unwrap());
    }
    Ok(out_seq)
}

/// Encodes a letter sequence, i.e., a sequence of A, C, T, and Gs,
/// to a one-hot encoded array of shape (4,L), where L is the length
/// of seq.
pub fn letter_seq_to_one_hot(seq: &str) -> Result<ndarray::Array2<u64>, String> {
    let lut: BTreeMap<char,usize> = BTreeMap::from([
        ('A', 0),
        ('C', 1),
        ('G', 2),
        ('T', 3),
    ]);

    let mut out_arr = ndarray::Array::zeros((4,seq.len()));
    for (c_idx,base) in seq.to_uppercase().char_indices() {
        let r_idx = lut.get(&base);
        if r_idx.is_none() {
            return Err(format!(
                "Key '{}' not found in lut {:?}",
                &base,
                &lut,
            ));
        };
        out_arr[[*r_idx.unwrap(), c_idx]] = 1;
    }
    Ok(out_arr)
}

pub fn seq_hamming_distance(seqA: &str, seqB: &str) -> Result<u64, Box<dyn Error>> {
    let lenA = seqA.len();
    let lenB = seqB.len();
    assert_eq!(lenA, lenB);
    let matA = letter_seq_to_one_hot(seqA)?;
    let matB = letter_seq_to_one_hot(seqB)?;
    Ok(hamming_distance(&matA.view(), &matB.view()))
}

/// Computes the Hamming distance between two letter sequences
pub fn hamming_distance(matA: &ndarray::ArrayView::<u64, Ix2>,
                        matB: &ndarray::ArrayView::<u64, Ix2>) -> u64 {
    let xor = matA ^ matB;
    // divide sum of XOR array by 2,
    // since there will be two 1's in each mismatched column
    xor.sum() / 2
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
/// - `weights` - a reference to a  view of a 2D array, typically a [Motif] `weights` field
pub fn stranded_weighted_manhattan_distance(
    arr1: &ndarray::ArrayView::<f64, Ix3>, 
    arr2: &ndarray::ArrayView::<f64, Ix2>,
    weights: &ndarray::ArrayView::<f64, Ix2>,
) -> ndarray::Array1<f64> {
    // This approach is much, much faster than the broadcasted ndarray
    // approach I used to have here. The old way was allocating new
    // arrays, so I'm guessing that's where the time was being spent.
    let fwd_diff = ndarray::Zip::from(arr1.slice(s![..,..,0]))
        .and(arr2)
        .and(weights)
        .fold(0.0, |acc, a, b, c| acc + (a-b).abs()*c);

    let rev_diff = ndarray::Zip::from(arr1.slice(s![..,..,1]))
        .and(arr2)
        .and(weights)
        .fold(0.0, |acc, a, b, c| acc + (a-b).abs()*c);

    ndarray::array![fwd_diff, rev_diff]
}

pub fn weighted_manhattan_distance(
    arr1: &ndarray::ArrayView::<f64, Ix2>, 
    arr2: &ndarray::ArrayView::<f64, Ix2>,
    weights: &ndarray::ArrayView::<f64, Ix2>
) -> f64 {
    ndarray::Zip::from(arr1)
        .and(arr2)
        .and(weights)
        .fold(0.0, |acc, a, b, c| acc + (a-b).abs()*c)
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


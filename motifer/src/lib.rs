use std::error::Error;
use ndarray::prelude::*;
use ndarray::Array;
// ndarray_stats exposes ArrayBase to useful methods for descriptive stats like min.
use ndarray_stats::QuantileExt;

// Next major goal:
//   Run seed over entire database of seqs and get MI.
//   + subgoal: benchmark testing for this!
//   + subgoal: get RecordsDatabase struct, with sequences field and y-vals field
//
//
// TODO: 
//
// 1. Generate hits vector for a given seed (see pseudocode below)
//    + We need the following:
//        + functions to increment the hit count and toggle maxed to True
//           when we hit the strand's max count
//        + method to slice backward through a sequence (or maybe we just compute the reverse at the beginning, along with the reversed weights? that might make it easier to start with all our data, and not have to make unnecessary copies)
//    + Where to parallelize is an engineering question
//        + just keep in mind for future, once we actually implemente par
//
//
//     rev_seq = seed_seq.rev
//     seed_seq.constrain_norm_weights(alpha)
//     rev_seq.constrain_norm_weights(alpha)
//
//     seed_hits = array(2d, first axis is record number, second axis is strand)
//
//     // loop over all references
//     for i,sequence in all_seqs.enumerate() {
//
//         rev_maxed = False
//         fwd_maxed = False
//       
//         // we probably started with 60-mer, but it's now 56 due to clipping 2 from each end
//         for window in reference.window_iter(0,56,15) {
//
//             if rev_maxed and fwd_maxed {
//                 break
//             }
//
//   ############################################################
//   ############################################################
//   ### make the distance/matching stuff an impl to SequenceView?
//   ############################################################
//   ############################################################
//             if not fwd_maxed {
//                 distance = weighted_manhattan_distance(seed_seq, window_seq, seed_seq.norm_weights)
//                 if distance < threshold {
//                     seed_hits[i,0] += 1
//                     if seed_hits[i,0] == max_count {
//                         fwd_maxed = True
//                     }
//                 }
//             }
//
//             if not rev_maxed {
//                 distance = weighted_manhattan_distance(rev_seq, window_seq, rev_seq.norm_weights)
//                 if distance < threshold {
//                     seed_hits[i,1] += 1
//                     if seed_hits[i,1] == max_count {
//                         rev_maxed = True
//                     }
//                 }
//             }
//         }
//     }

#[cfg(test)]
mod tests {
    use super::*;

    fn set_up_motif(val: f64, size: usize) -> Sequence {
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
        let this_seq = set_up_motif(2.0, 30);
        for window in this_seq.window_iter(0, 10, 3) {
            assert_eq!(Array2::from_elem((5, 3), 2.0),
                       window.params)
        }
    }
    
    #[test]
    fn test_pairwise_comparison(){
        let this_seq = set_up_motif(2.0, 30);
        let mut out = Vec::new();
        for window1 in this_seq.window_iter(0, 31, 3) {
            for window2 in this_seq.window_iter(0, 31, 3) {
                out.push(manhattan_distance(&window1.params, &window2.params));
            }
        }
        println!("{:?}", out);
        assert_eq!(out.len(), (30-2)*(30-2));
    }

    #[test]
    fn test_motif_normalization() {
        let this_seq = set_up_motif(2.0, 31);
        let mut this_motif: Motif = this_seq
            .window_iter(0, 31, 12)
            .next()
            .unwrap()
            .into();
        // check that it initializes to 1
        assert_eq!(this_motif.weights.weights.sum(), 1.0*12.0*5.0);
        // check that normed weights intialize to 0
        assert_eq!(this_motif.weights.weights_norm.sum(), 0.0);
        this_motif.weights.normalize();
        // check that when normalized weights are basically 1
        let sum_normed = this_motif.weights.weights_norm.sum();
        assert!((sum_normed > 0.99999) && (sum_normed < 1.00001));
        // check that when normalized. Unnormalized weights are untouched
        assert_eq!(this_motif.weights.weights.sum(), 1.0*12.0*5.0);
    }

    #[test]
    fn test_pairwise_motif() {
        let alpha = 0.1;
        let this_seq = set_up_motif(2.0, 30);
        let mut out = Vec::new();
        for seed in this_seq.window_iter(0, 31, 3){
            // get the motif weights through type conversion
            let mut this_motif: Motif = seed.into();
            // normalize the weights before using them
            this_motif.weights.constrain_normalize(&alpha);
            for window in this_seq.window_iter(0, 31, 3){
                out.push(
                    weighted_manhattan_distance(
                        // I don't like how much we have to access internal
                        // struct fields here but we can revisit
                        &this_motif.params.params, 
                        &window.params, 
                        // use normalized weights. Need to explicitly get a view
                        &this_motif.weights.weights_norm.view()
                    )
                );
            }
        }
        println!("{:?}", out);
        assert_eq!(out.len(), (30-2)*(30-2));
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
        assert!((sum_normed > 0.99999) && (sum_normed < 1.00001));
        assert!(motif_weights.weights_norm.abs_diff_eq(&target_arr, 1e-6));
    }

    #[test]
    fn test_RecordsDB_seq_iter(){
        let this_seq = set_up_motif(2.0, 32);
        let this_seq2 = set_up_motif(3.0, 20);
        let this_db = RecordsDB::new(vec![this_seq, this_seq2], array![0.0,1.0]);
        println!("{:?}", this_db)
    }

}

//fn run_query_over_ref(
//    y_vals: Vec<f64>,
//    query_shapes: Array<f64>,
//    query_weights: Array<f64>,
//    threshold: f64,
//    reference: Array<f64>,
//    R: u32,
//    W: u32,
//    dist_func: &dyn Fn(Array<f64>, Array<f64>, Array<f64>) -> f64,
//    max_count: u32,
//) {
//}


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
#[derive(Debug)]
pub struct Sequence {
    params: ndarray::Array2<f64>
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
#[derive(Debug)]
pub struct SequenceView<'a> {
    params: ndarray::ArrayView::<'a, f64, Ix2>
}

/// Represents a motif as a [SequenceView] with associated [MotifWeights]
///
/// # Fields
///
/// * `params` - Stores the sequence values as a [SequenceView]
/// * `weights` - Stores the associated weights as a [MotifWeights]
#[derive(Debug)]
pub struct Motif<'a> {
    params: SequenceView<'a>,
    weights: MotifWeights,
}

/// Represents the weights for a [Motif] in it's own structure
///
/// # Fields
///
/// * `weights` - Stores the weights as a 2d array
/// * `weights_norm` - Caches normalized weights as needed
#[derive(Debug)]
pub struct MotifWeights {
    weights: ndarray::Array2::<f64>,
    weights_norm: ndarray::Array2::<f64>,
}

/// Represents a database of Sequences and their associated value
///
/// # Fields
///
/// * `seqs` - Stores [Sequence] classes in a vector
/// * `values` - Stores associated values in a vector in 1D array
#[derive(Debug)]
pub struct RecordsDB {
    seqs: Vec<Sequence>,
    values: ndarray::Array1::<f64>
}


//NOTE: do we know that param types will always be in the same order?
//  or do we need to set up a hashmap to assign params to
//  the appropriate row in arr based on the index we get by hashing
//  by param name?
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
impl<'a> Iterator for SequenceIter<'a> {
    type Item = SequenceView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let this_start = self.start;
        let this_end = self.start + self.size;
        if this_end == self.end{
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

/// This allows the syntatic sugar of `for val in param` to work
impl<'a> IntoIterator for &'a Param {
    type Item = &'a f64;
    type IntoIter = ndarray::iter::Iter<'a, f64, Ix1>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> Motif<'a> {
    /// Returns a new motif instance by bundling with a weight vector
    /// of [MotifWeights] type. 
    ///
    /// # Arguments
    ///
    /// * `params` - a [SequenceView] that defines the motif
    pub fn new(params: SequenceView<'a>) -> Motif {
        let weights = MotifWeights::new(&params);
        Motif{params, weights}
    }
}

/// Allow conversion from a [SequenceView] to a [Motif]
impl<'a> From<SequenceView<'a>> for Motif<'a> {
    fn from(sv : SequenceView<'a>) -> Motif<'a>{
        Motif::new(sv)
    }
}


impl MotifWeights {
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

impl RecordsDB {

    pub fn new(seqs: Vec<Sequence>, values: ndarray::Array1::<f64>) -> RecordsDB {
        RecordsDB{seqs, values}
    }
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
/// - `arr1` - a reference to a view of a 2D array, typically a [Motif] `param` field
/// - `arr2` - a reference to a view of a 2D array, typically a window on a sequence to be compared
/// - `weights` - a view of a 2D array, typically a [Motif] `weights` field
pub fn weighted_manhattan_distance(arr1: &ndarray::ArrayView::<f64, Ix2>, 
                                   arr2: &ndarray::ArrayView::<f64, Ix2>,
                                   weights: &ndarray::ArrayView::<f64, Ix2>) -> f64 {
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

//fn parse_all_args(mat: clap::ArgMatches) -> Args {
//    
//    let kmer = fetch_int_arg(
//        mat
//        .value_of("kmer")
//        .unwrap()
//    );
//
//    let thresh_perc = fetch_float_arg(
//        mat
//        .value_of("threshold_perc")
//        .unwrap()
//    );
//
//    let thresh_mat = fetch_float_arg(
//        mat
//        .value_of("threshold_perc")
//        .unwrap()
//    );
//
//    let cont = fetch_int_arg(
//        mat
//        .value_of("continuous")
//        .unwrap()
//    );
//
//    let inf_r = fetch_int_arg(
//        mat
//        .value_of("inforobust")
//        .unwrap()
//    );
//
//    let fr_jack = fetch_float_arg(
//        mat
//        .value_of("frackjack")
//        .unwrap()
//    );
//
//    let args = Args {
//        infile: mat.value_of("infile").unwrap().to_string(),
//        // NOTE: unwrap is not safe for args that are not required.
//        //  should do if let for args that are not required.
//        //param_files: mat.values_of("params").unwrap().collect(),
//        //param_names: mat.values_of("param_names").unwrap().collect(),
//        kmer: kmer,
//        threshold_perc: thresh_perc,
//        threshold_match: thresh_mat,
//        continuous: cont,
//        inforobust: inf_r,
//        fracjack: fr_jack,
//        out_pref: mat.value_of("out_pref").unwrap(),
//    };
//
//    return args
//}

///// Parses a CLI match from clap crate and converts to u32
//pub fn fetch_int_arg(arg_val: &str) -> u32 {
//    let int_val: u32 = arg_val.parse().unwrap();
//    return int_val
//}
//
///// Parses a CLI match from clap crate and converst to f64
//pub fn fetch_float_arg(arg_val: &str) -> f64 {
//    let float_val: f64 = arg_val.parse().unwrap();
//    return float_val
//}

//fn parse_cli_yaml(cfg_fname: &str) -> Args {
//
//    // get yaml file containing argument structure
//    let yaml = clap::load_yaml!(cfg_fname);
//    let matches = clap::App::from_yaml(yaml).get_matches();
//
//    let args = parse_all_args(matches);
//
//    return args
//}

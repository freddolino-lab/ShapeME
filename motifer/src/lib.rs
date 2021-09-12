use std::error::Error;
use ndarray::prelude::*;
use ndarray::Array;

#[cfg(test)]
mod tests {
    use super::*;

    fn set_up_motif(val: f32, size: usize) -> Sequence {
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
    fn test_const_dist() {
        let this_motif = set_up_motif(2.0, 10);
        let that_motif = set_up_motif(1.0, 10);
        assert_eq!(manhattan_distance(&this_motif, &that_motif), 1.0*10.0*5.0);
    }
    #[test]
    fn test_window_over_param() {
        let param = Param::new(ParamType::EP, Array::<f32, _>::linspace(0.0, 20.0, 21)).unwrap();
        let mut i = 0.0;
        for window in param.windows(5){
            assert_eq!(window, Array::<f32, _>::linspace(i, i + 4.0, 5));
            i += 1.0;
        }
    }

    #[test]
    fn test_window_over_seq(){
        let this_motif = set_up_motif(2.0, 10);
        for window in this_motif.window_iter(0, 10, 3){
            assert_eq!(vec![array![2.0, 2.0, 2.0]; 5],
                       window.params)
        }
    }
    
    #[test]
    fn test_pairwise_comparison(){
        let this_seq = set_up_motif(2.0, 30);
        let mut out = Vec::new();
        for window1 in this_seq.window_iter(0, 31, 3){
            for window2 in this_seq.window_iter(0, 31, 3){
                out.push(manhattan_distance2(&window1, &window2));
            }
        }
        println!("{:?}", out);
        assert_eq!(2, 1);
    }
 
}

//fn run_query_over_ref(
//    y_vals: Vec<f32>,
//    query_shapes: Array<f32>,
//    query_weights: Array<f32>,
//    threshold: f32,
//    reference: Array<f32>,
//    R: u32,
//    W: u32,
//    dist_func: &dyn Fn(Array<f32>, Array<f32>, Array<f32>) -> f32,
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
#[derive(Debug)]
pub struct Param {
    /// Must have an associated name of type [ParamType]
    name: ParamType,
    /// Values are stored as a 1 dimensional ndarray of floating point numbers
    vals: Array::<f32,Ix1>,
}

/// Represents a single sequence as a combination of [Param] objects
#[derive(Debug)]
pub struct Sequence {
    params: Vec<Param>
}

/// Represents the state needed for windowed iteration over a [Sequence]
pub struct SequenceIter<'a>{
    /// Stores an initial start to the iteration
    start: usize,
    /// The ending point of the iteration
    end: usize,
    /// The size of the window to iterate over
    size: usize,
    /// A reference to the [Sequence] that is getting iterated over
    sequence: &'a Sequence
}


/// Represents an unmutable windowed view to a [Sequence]
#[derive(Debug)]
pub struct SequenceView<'a> {
    /// The view is stored as a vector of 1 [ndarray::ArrayView] for each
    /// parameter in the sequence
    params: Vec<ndarray::ArrayView::<'a, f32, Ix1>>
}

/// For Motif, the idea here is that info has a key for each parameter.
///  The value associated with each parameter is a vector of tuples.
///  The first element of each tuple is the parameter's value, the second
///  is the weight.
pub struct Motif<'a> {
    params: SequenceView<'a>,
    weights: MotifWeights,
}

/// Store the weights for a motif in it's own structure
pub struct MotifWeights {
    weights: Vec<Array::<f32, Ix1>>
}

///// Record contains a single piece of DNA's shape values and y-value
/////  Also has a windows attribute, which is a vector of the windows
/////  for the split up parameter values.
//#[derive(Debug)]
//struct Record {
//    windows: Vec<HashMap>,
//    y: u8,
//}

impl Sequence {
    /// Returns a Result containing a new sequence or any errors that 
    /// occur in attempting to create it.
    ///
    /// # Arguments
    ///
    /// * `params` - A vector of [Param] objects
    pub fn new(params: Vec<Param>) -> Result<Sequence, Box<dyn Error>> {
        Ok(Sequence{ params })
    }

    /// Creates a read-only windowed iterator over the sequence. Automatically
    /// slides by 1 unit.
    ///
    /// # Arguments
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
    /// * `params` - a vector of ndarray slices representing a subset of the given sequence
    pub fn new(params: Vec<ndarray::ArrayView::<'a,f32, Ix1>>) -> SequenceView<'a> {
        SequenceView { params }
    }

    pub fn iter(&self) -> core::slice::Iter<ndarray::ArrayView::<'a, f32, Ix1>>{
        self.params.iter()
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
            let mut out = Vec::new();
            for param in self.sequence.params.iter() {
                out.push(param.vals.slice(s![this_start..this_end]));
            }
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
    pub fn new(name: ParamType, vals: Array::<f32, Ix1>) -> Result<Param, Box<dyn Error>> {
        Ok(Param {name, vals})
    }

    /// Enables subtraction between two full parameter vectors without
    /// direct access to the inner values.
    ///
    /// Returns the element-wise subtraction.
    pub fn subtract(&self, other: &Param) -> Vec<f32> {
        if self.name != other.name {
            panic!("Can't subtract params of different types")
        } else {
            self.iter().zip(other).map(|(a, b)| a - b).collect()
        }
    }
    /// Returns an iterator over the individual values in the inner array
    fn iter(&self) -> ndarray::iter::Iter<f32, Ix1> {
        self.vals.iter()
    }
    /// Returns a windowed iterator over the inner values array
    ///
    /// # Arguments
    ///
    /// * `size` - size of the window to iterate over
    fn windows(&self, size: usize) -> ndarray::iter::Windows<f32, Ix1>{
        self.vals.windows(size)
    }
}



/// This allows the syntatic sugar of `for val in param` to work
impl<'a> IntoIterator for &'a Param {
    type Item = &'a f32;
    type IntoIter = ndarray::iter::Iter<'a, f32, Ix1>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> Motif<'a> {

    pub fn new(params: SequenceView<'a>) -> Motif {
        let weights = MotifWeights::new(&params);
        Motif{params, weights}
    }
}

impl MotifWeights {
    pub fn new(params: &SequenceView) -> MotifWeights {
        let mut weights = Vec::new();
        for val in params.iter(){
            weights.push(Array::ones(val.len()));
        }
        MotifWeights{weights}
    }
}



pub fn manhattan_distance(seq1: &Sequence, seq2: &Sequence) -> f32 {
    let mut distance: f32 = 0.0;
    for (p1, p2) in seq1.params.iter().zip(&seq2.params){
        distance += p1.subtract(&p2).iter().map(|x| x.abs()).sum::<f32>();
    }
    distance
}

pub fn manhattan_distance2(seq1: &SequenceView, seq2: &SequenceView) -> f32 {
    let mut distance: f32 = 0.0;
    for (p1, p2) in seq1.params.iter().zip(&seq2.params){
        distance += (p1 - p2).iter().map(|x| x.abs()).sum::<f32>();
    }
    distance
}

//pub fn constrained_manhattan_distance(seq1: &SequenceView, seq2: &SequenceView,
//                                      weights: &MotifWeights) -> f32 {
//    let mut distance: f32 = 0.0;
//    let norm_weights = weights.iter()
//}

        


 //impl Motif {
 //    /// Returns weighted distance between two Motif structs.
 //    ///
 //    /// # Examples
 //    ///
 //    /// ```
 //    /// ```
 //    fn constrained_manhattan_distance(
 //        &self,
 //        other: &Motif,
 //    ) -> f32 {
 //
 //        let mut dist: f32 = 0.0;
 //
 //        let mut dist_vec = Vec::new();
 //        //NOTE: needs type here
 //        let mut self_vals = Vec::new();
 //        let mut other_vals = Vec::new();
 //        let mut w_exp_vec = Vec::new();
 //        let mut w_exp_sum = 0;
 //
 //        for (param_name,param) in self.params {
 //            // normalize the weights here
 //            
 //            for pos in 0..param.vals.len() {
 //                let self_val = param.vals[pos];
 //                let w = param.weights[pos];
 //                let w_exp = math::exp(w);
 //                w_exp_vec.push(w_exp);
 //                w_exp_sum += w_exp;
 //
 //                let other_val = other.params.get(&param_name).vals[pos];
 //                let dist = math::abs(self_val - other_val);
 //                dist_vec.push(dist);
 //            }
 //        }
 //        
 //        for idx in 0..dist_vec.len() {
 //            let w_exp = w_exp_vec[idx] / w_exp_sum;
 //            dist += w_exp * dist_vec[idx];
 //        }
 //
 //        return dist
 //    }
 //}

// NOTE: we can start from this to implement subtraction of two motifs
// I'm basically thinking we have `sub` be a method of a struct, `Motif`.
// We wouldn't have the output be another `Motif` struct, as defined here,
// but at least this gives us a start.
//impl Sub for Motif {
//
//    fn sub(self, other: &Motif) -> f64 {
//        for (param_name, param_vals) in other.vals {
//            let self_vals = self.vals.get(&param_name)
//            let dist = // implement Sub for params
//        }
//    }
//}

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
///// Parses a CLI match from clap crate and converst to f32
//pub fn fetch_float_arg(arg_val: &str) -> f32 {
//    let float_val: f32 = arg_val.parse().unwrap();
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
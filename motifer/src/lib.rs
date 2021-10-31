use std::error::Error;
use ndarray::prelude::*;
use ndarray::Array;
// ndarray_stats exposes ArrayBase to useful methods for descriptive stats like min.
use ndarray_stats::QuantileExt;

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
        let this_seq = set_up_sequence(2.0, 31);
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
        // check that when normalized, weights sum to 1
        let sum_normed = this_motif.weights.weights_norm.sum();
        assert!(AbsDiff::default().epsilon(1e-6).eq(&sum_normed, &1.0));
        // check that when normalized. Unnormalized weights are untouched
        assert_eq!(this_motif.weights.weights.sum(), 1.0*12.0*5.0);
    }

    #[test]
    fn test_pairwise_motif() {
        let alpha = 0.1;
        let this_seq = set_up_sequence(2.0, 30);
        let mut out = Vec::new();
        for seed in this_seq.window_iter(0, 31, 3){
            // get the motif weights through type conversion
            let mut this_motif: Motif = seed.into();
            // normalize the weights before using them
            this_motif.normalize_weights(&alpha);
            for window in this_seq.window_iter(0, 31, 3){
                let dist = this_motif.distance(&window);
                out.push(dist);
            }
        }
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
        assert!(AbsDiff::default().epsilon(1e-6).eq(&sum_normed, &1.0));
        assert!(motif_weights.weights_norm.abs_diff_eq(&target_arr, 1e-6));
    }

    #[test]
    fn test_RecordsDB_seq_iter(){
        let this_seq = set_up_sequence(2.0, 32);
        let this_seq2 = set_up_sequence(3.0, 20);
        let this_db = RecordsDB::new(vec![this_seq, this_seq2], array![0.0,1.0]);
        for entry in this_db.iter(){
            println!("{:?}", entry);
        }
    }

    #[test]
    fn test_get_seeds() {
        let kmer = 15;
        let this_seq = set_up_sequence(2.0, 30);
        let that_seq = set_up_sequence(2.0, 60);
        let rec_db = RecordsDB::new(vec![this_seq, that_seq], array![0.0,1.0]);
        let seeds = rec_db.make_seed_vec(kmer, 0.01);
        assert_eq!(seeds.seeds.len(), 60)
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
            threshold1,
            max_count,
        );
        assert_eq!(hits[0], 0);
        assert_eq!(hits[1], 0);

        let hits = that_seq.count_hits_in_seq(
            &seed.params.params,
            &wv,
            threshold2,
            max_count,
        );
        assert_eq!(hits[0], 2);
        assert_eq!(hits[1], 0);
    }

    fn setup_RecordsDB(num_seqs: usize, length_seqs: usize) -> RecordsDB{
        let mut seqs = Vec::new();
        let mut vals = Vec::new();
        for i in 0..num_seqs{
            seqs.push(set_up_sequence(1.0 + i as f64, length_seqs));
            vals.push(i as f64);
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
                    1.0,
                    10);
        assert_eq!(hits[[6,0]], 10)
    }


}

//pub fn run_query_over_refs() {
//
//    // iterate over records
//    for r in range(R):
//
//        // get the reference sequence from records db
//        this_ref = rec_db[r,:,:,:,:]
//        
//        count_hits_in_seq()
//}


// MODIFY TO ACCEPT FLEXIBLE TYPE FOR query argument; could be EITHER Motif OR Seed
///// For a single Motif, count the number of times its distance to a window
///// of a Sequence falls below the specified threshold, i.e., matches the
///// Sequence.
/////
///// # Arguments
/////
///// * `query` - a Seed, the sequence of which will be compared to each window in seq.
///// * `seq` - reference Sequence to which query is being compared
///// * `kmer` - window length (might not be necessary, since we can grab this from the seed directly
///// * `threshold` - distance between the query and seq below which a hit is called
///// * `max_count` - maximum number of times a hit will be counted on each strand
//pub fn count_hits_in_seq(query: &Seed, seq: &Sequence,
//                         kmer: usize,
//                         threshold: f64, max_count: i64)
//    -> ndarray::Array<i64, Ix1> {
//
//    // set maxed to false for each strand
//    let mut f_maxed = false;
//    //////////////////////////////////
//    // SET TO TRUE FOR REVERSE UNTIL WE ACTUALLY START USING STRANDEDNESS
//    //////////////////////////////////
//    let mut r_maxed = true;
//    let mut hits = ndarray::Array::zeros(2);
//
//    // iterate through windows of seq
//    for window in seq.window_iter(0, seq.params.dim().1, kmer) {
//
//        // once both strands are maxed out, stop doing comparisons
//        if f_maxed & r_maxed {
//            break
//        }
//        // get the distance.
//        /////////////////////////////////////////////////
//        // IN THE FUTURE, WE'LL BROADCAST TO BOTH STRANDS
//        /////////////////////////////////////////////////
//        let dist = query.distance(&window);
//        /////////////////////////////////////////////////
//        // ONCE WE'VE IMPLEMENTED STRANDEDNESS, SLICE APPROPRIATE STRAND'S DISTANCE HERE
//        /////////////////////////////////////////////////
//        if (dist < threshold) & (!f_maxed) {
//            hits[0] += 1;
//            if hits[0] == max_count {
//                f_maxed = true;
//            }
//        } 
//        /////////////////////////////////////////////////
//        /////////////////////////////////////////////////
//        /////////////////////////////////////////////////
//        //if (dist[1] < threshold) & (!r_maxed) {
//        //    hits[1] += 1;
//        //    if hits[1] == max_count {
//        //        r_maxed = true;
//        //    }
//        //} 
//
//    }
//    // return the hits
//    hits
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
/// * `threshold` - Stores the associated threshold for what is
///                 considered a match.
#[derive(Debug)]
pub struct Motif<'a> {
    // NOTE: we'll need to make this its own Sequence so that we can update
    //  shapes as well as weights during optimization.
    params: SequenceView<'a>,
    weights: MotifWeights, // the MotifWeights struct contains two 2D-Arrays.
    threshold: f64,
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

// My goal for this struct is to be able to have a way to use a read-only pointer
//  to each window of sequences using a SequenceView, and to have all Seeds point
//  to the same set of weights using the ArrayView.
// This way we can calculate all our initial MIs without copying unnecessarily.
// After filtering by CMI we can then create Motif structs that each own their
//  shapes and weights. 
#[derive(Debug)]
pub struct Seed<'a> {
    params: SequenceView<'a>,
    hits: ndarray::Array2::<f64>,
    mi: f64,
}
#[derive(Debug)]
// We have to create a container to hold the weights with the seeds
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
    seqs: Vec<Sequence>,
    values: ndarray::Array1::<f64>
}

impl<'a> Seed<'a> {
    pub fn new(params: SequenceView<'a>,
               record_num: usize) -> Seed<'a> {
        let hits = ndarray::Array2::zeros((record_num, 2));
        let mi = 0.0;
        Seed{params, hits, mi}
    }

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

/// Stores a single entry of the RecordsDB 
/// # Fields
///
/// * `seq` - A reference to a [Sequence] classe
/// * `value` - The associated value for the sequence
#[derive(Debug)]
pub struct RecordsDBEntry<'a> {
    seq: &'a Sequence,
    value: f64
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

    // MODIFY TO ACCEPT FLEXIBLE TYPE FOR query argument; could be EITHER Motif OR Seed
    // NOTE: Rust can't do flexible types. Instead wrote it to take only what
    // it needs rather than a full object.
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
                             threshold: f64, max_count: i64) -> Array<i64, Ix1> {
    
        // set maxed to false for each strand
        let mut f_maxed = false;
        //////////////////////////////////
        // SET TO TRUE FOR REVERSE UNTIL WE ACTUALLY START USING STRANDEDNESS
        //////////////////////////////////
        let mut r_maxed = true;
        let mut hits = ndarray::Array::zeros(2);
    
        // iterate through windows of seq
        for window in self.window_iter(0, self.seq_len(), query.raw_dim()[1]) {
    
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
            if (dist < threshold) & (!f_maxed) {
                hits[0] += 1;
                if hits[0] == max_count {
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

/// This allows the syntactic sugar of `for val in param` to work
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
    pub fn new(params: SequenceView<'a>, threshold: f64) -> Motif {
        let weights = MotifWeights::new(&params);
        Motif{params, weights, threshold}
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
            &self.params.params, 
            &ref_seq.params, 
            &self.weights.weights_norm.view()
        )
    }
}


/// Allow conversion from a [SequenceView] to a [Motif]
impl<'a> From<SequenceView<'a>> for Motif<'a> {
    fn from(sv : SequenceView<'a>) -> Motif<'a>{
        Motif::new(sv, 0.0)
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

    /// Returns a new RecordsDB holding sequence value pairs as separate
    /// vectors.
    ///
    /// # Arguments
    ///
    /// * `seqs` - a vector of [Sequence]
    /// * `values` - a vector of values for each sequence
    pub fn new(seqs: Vec<Sequence>, values: ndarray::Array1::<f64>) -> RecordsDB {
        RecordsDB{seqs, values}
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
            for window in entry.seq.window_iter(0, entry.seq.seq_len(), kmer) {
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

    /// Iterate over each record in the database as a [Sequence] value pair
    pub fn iter(&self) -> RecordsDBIter{
        RecordsDBIter{loc: 0, db: &self, size: self.seqs.len()}
    }

    pub fn get_hits(&self, query: &ndarray::ArrayView<f64, Ix2>,
                  weights: &ndarray::ArrayView<f64, Ix2>,
                  threshold: f64, max_count: i64) -> Array<i64, Ix2> {

        let mut hits = ndarray::Array2::zeros((self.len(), 2));
        for (i, entry) in self.iter().enumerate(){
            let this_hit = entry.seq.count_hits_in_seq(query, weights,
                                                       threshold, max_count);
            hits.row_mut(i).assign(&this_hit);
        }
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
    pub fn new(seq: &Sequence, value: f64) -> RecordsDBEntry {
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

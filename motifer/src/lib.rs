use std::error::Error;
use std::cmp;
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
    fn test_dot_prod() {
        // first element is always 1, second element is max_count+1
        let a = arr1(&[1, 3]);   // fwt dot prod | rev dot prod
        // example hits array
        let b = arr2(&[[0, 0],   // 0            | 0
                       [1, 0],   // 1            | 3
                       [2, 0],   // 2            | 6
                       [0, 1],   // 3            | 1
                       [1, 1],   // 4            | 4
                       [2, 1],   // 5            | 7
                       [0, 2],   // 6            | 2
                       [1, 2],   // 7            | 5
                       [2, 2]]); // 8            | 8
        let dot_prod = b.dot(&a);
        println!("{}", dot_prod);
        let rev_prod = b.slice(s![.., ..;-1]).dot(&a);
        println!("{}", rev_prod);
        assert_eq!(arr1(&[0, 1, 2, 3, 4, 5, 6, 7, 8]), dot_prod);
        assert_eq!(arr1(&[0, 3, 6, 1, 4, 7, 2, 5, 8]), rev_prod);
    }

    #[test]
    fn sort_hits() {
        let answer = arr1(&[0, 3, 6, 3, 4, 7, 6, 7, 8]);
        let max_count = 2;
        // first element is always 1, second element is max_count+1
        let a = arr1(&[1, max_count + 1]); 
        // example hits array        // fwt dot prod | rev dot prod
        let mut b = arr2(&[[0, 0],   // 0            | 0
                           [1, 0],   // 1            | 3
                           [2, 0],   // 2            | 6
                           [0, 1],   // 3            | 1
                           [1, 1],   // 4            | 4
                           [2, 1],   // 5            | 7
                           [0, 2],   // 6            | 2
                           [1, 2],   // 7            | 5
                           [2, 2]]); // 8            | 8
        let min = b.map_axis(ndarray::Axis(1), |r| cmp::min(r[0], r[1]));
        let max = b.map_axis(ndarray::Axis(1), |r| cmp::max(r[0], r[1]));
        
        println!("{}", min);
        println!("{}", max);

        b.column_mut(0).assign(&min);
        b.column_mut(1).assign(&max);

        println!("{}", b);

        let categories = b.dot(&a);

        println!("{}", categories);
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
        println!("{:?}", out);
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

    //#[test]
    //fn test_get_seeds() {
    //    let kmer = 15;
    //    let this_seq = set_up_sequence(2.0, length);
    //    let that_seq = set_up_sequence(2.0, length);
    //    let rec_db = RecordsDB::new(vec![this_seq, that_seq], array![0.0,1.0]);
    //    let mut seeds = rec_db.make_seed_vec(kmer);
    //    println!("{:?}", seeds)
    //}

    #[test]
    fn test_hit_counting(){
        let kmer = 15;
        let threshold1 = 0.0;
        let threshold2 = 1.0;
        let max_count = 2;
        let alpha = 0.01;
        let length = 30;
        let this_seq = set_up_sequence(2.0, kmer);
        let that_seq = set_up_sequence(2.1, length);
        let that_view = this_seq.view();

        let mut seed_weights = MotifWeights::new(&that_view);
        seed_weights.constrain_normalize(&alpha);
        let wv = seed_weights.weights_norm.view();

        let mut seed = Seed::new(&that_view, &wv, kmer);
        
        let hits = count_hits_in_seq(
            &seed,
            &that_seq,
            kmer,
            threshold1,
            max_count,
        );
        println!("Should have zero hits: {:?}", hits);

        let hits = count_hits_in_seq(
            &seed,
            &that_seq,
            kmer,
            threshold2,
            max_count,
        );
        println!("Should have two hits: {:?}", hits);


        //for seq in this_db {
        //    for window in seq.window_iter(0,length,kmer) {
        //        let mut seed = Seed::new(&window, &wv, rec_num);
        //        let hits = count_hits_in_seq(
        //            &seed,
        //            &seq,
        //            kmer,
        //            threshold,
        //            max_count,
        //        );
        //        hits_vec.push(hits);
        //        //println!("{:?}", hits);
        //        //seed.get_mi(
        //        //    &this_db,
        //        //    &kmer,
        //        //    &threshold,
        //        //    &max_count,
        //        //);
        //        //seed_vec.push(seed);
        //    }
        //}
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
/// For a single Motif, count the number of times its distance to a window
/// of a Sequence falls below the specified threshold, i.e., matches the
/// Sequence.
///
/// # Arguments
///
/// * `query` - a Seed, the sequence of which will be compared to each window in seq.
/// * `seq` - reference Sequence to which query is being compared
/// * `kmer` - window length (might not be necessary, since we can grab this from the seed directly
/// * `threshold` - distance between the query and seq below which a hit is called
/// * `max_count` - maximum number of times a hit will be counted on each strand
pub fn count_hits_in_seq(query: &Seed, seq: &Sequence,
                         kmer: usize,
                         threshold: f64, max_count: i64)
    -> ndarray::Array<i64, Ix1> {

    // set maxed to false for each strand
    let mut f_maxed = false;
    //////////////////////////////////
    // SET TO TRUE FOR REVERSE UNTIL WE ACTUALLY START USING STRANDEDNESS
    //////////////////////////////////
    let mut r_maxed = true;
    let mut hits = ndarray::Array::zeros(2);

    // iterate through windows of seq
    for window in seq.window_iter(0, seq.params.dim().1, kmer) {

        // once both strands are maxed out, stop doing comparisons
        if f_maxed & r_maxed {
            break
        }
        // get the distance.
        /////////////////////////////////////////////////
        // IN THE FUTURE, WE'LL BROADCAST TO BOTH STRANDS
        /////////////////////////////////////////////////
        let dist = query.distance(&window);
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
pub struct Seed<'a> {
    params: &'a SequenceView<'a>,
    weights: &'a ndarray::ArrayView::<'a, f64, Ix2>,
    hits: ndarray::Array2::<f64>,
    mi: f64,
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

impl<'a> Seed<'_> {
    pub fn new(params: &'a SequenceView<'a>,
               weights: &'a ndarray::ArrayView<'a, f64, Ix2>,
               record_num: usize) -> Seed<'a> {
        let hits = ndarray::Array2::zeros((record_num, 2));
        let mi = 0.0;
        Seed{params, weights, hits, mi}
    }

    pub fn distance(&self, ref_seq: &SequenceView) -> f64 {
        weighted_manhattan_distance(
            &self.params.params, 
            &ref_seq.params, 
            &self.weights
        )
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
    /// # Argumrnts
    ///
    /// * `kmer` - Length of windows to slice over each record's parameters
    /// * `alpha` - Lower limit on inv_logit transformed weights
    //pub fn make_seed_vec(&self, kmer: usize, alpha: f64) -> Vec<Seed> {

    //    let mut seed_vec = Vec::new();

    //    let r_num = self.seqs[0].params.raw_dim()[0];
    //    let c_num = self.seqs[0].params.raw_dim()[1];
    //    let const_seq = Sequence{params: ndarray::Array2::zeros((r_num, c_num))};
    //    let const_view = const_seq.view();

    //    let mut seed_weights = MotifWeights::new(&const_view);
    //    seed_weights.constrain_normalize(&alpha);
    //    let wv = &seed_weights.weights_norm.view();

    //    for entry in self.iter() {
    //        let seq = entry.seq;
    //        for window in seq.window_iter(0, self.len(), kmer) {
    //            let mut seed = Seed::new(&window, wv, self.len());
    //            seed_vec.push(seed);
    //        }
    //    }
    //    seed_vec
    //}

    /// Iterate over each record in the database as a [Sequence] value pair
    pub fn iter(&self) -> RecordsDBIter{
        RecordsDBIter{loc: 0, db: &self, size: self.seqs.len()}
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

use std::path;
use std::env;
use std::io::BufReader;
use std::fs::File;
use serde::Deserialize;
use ndarray::prelude::*;
use info_theory;

#[cfg(test)]
mod tests {
    use super::*;

    fn set_up_data() -> MotifsData {
        let y = vec![0,0,0,1,1,1];
        let hits_a = vec![0,0,0,1,1,1];
        let ami_a: f64 = 0.9;
        let param_num_a: usize = 30;
        let hits_b= vec![0,0,0,1,1,1];
        let ami_b: f64 = 0.7;
        let param_num_b: usize = 30;
        let hits_c= vec![0,0,0,1,1,1];
        let ami_c: f64 = 0.0;
        let param_num_c: usize = 30;
        let hits_d= vec![0,0,0,2,2,2];
        let ami_d: f64 = 0.9;
        let param_num_d: usize = 30;

        let rec_num: u64 = 100;

        let motif_a = MotifData{
            hits: hits_a,
            ami: ami_a,
            param_num: param_num_a,
        };
        let motif_b = MotifData{
            hits: hits_b,
            ami: ami_b,
            param_num: param_num_b,
        };
        let motif_c = MotifData{
            hits: hits_c,
            ami: ami_c,
            param_num: param_num_c,
        };
        let motif_d = MotifData{
            hits: hits_d,
            ami: ami_d,
            param_num: param_num_d,
        };
        let motifs = MotifsData{
            y: y.to_vec(),
            motifs: vec![motif_a, motif_b, motif_c, motif_d],
            rec_num:rec_num,
        };
        motifs
    }

    #[test]
    fn test_filter() {
        let motifs_data = set_up_data();
        let inds = motifs_data.get_good_motif_indices();
        println!("{:?}", inds);
    }
}


#[derive(Deserialize)]
struct MotifsData {
    y: Vec<i64>,
    motifs: Vec<MotifData>,
    rec_num: u64,
}

#[derive(Deserialize)]
struct MotifData {
    hits: Vec<i64>,
    ami: f64,
    param_num: usize,
}

impl MotifsData {
    fn get_good_motif_indices(&self) -> String {

        // store indices of retained motifs
        let mut retained_inds: Vec<usize> = Vec::new();
        let mut result = String::new();
        let y_vals = Array::from_vec(self.y.to_vec());

        // Make sure first seed passes AIC
        let log_lik = self.rec_num as f64 * self.motifs[0].ami;
        let mut model_param_num: usize = self.motifs[0].param_num;

        // get aic, if it's less than zero, insert the index into the vec
        let aic = info_theory::calc_aic(model_param_num, log_lik);
        if aic < 0.0 {
            retained_inds.push(0);
            let res = 0.to_string();
            result.push_str(&res);
        } else {
            return String::new();
        }

        // loop through candidate motifs
        for (i,cand_motif) in self.motifs[1..self.motifs.len()].iter().enumerate() {
            let ind = i+1;
            //println!("index: {}", ind);

            // if this motif doesn't pass AIC on its own, with number of params, skip it
            let log_lik = self.rec_num as f64 * cand_motif.ami;
            if info_theory::calc_aic(cand_motif.param_num, log_lik) > 0.0 {
                continue
            }

            let cand_cats = Array::from_vec(cand_motif.hits.to_vec());
            let mut motif_pass = true;

            for motif_ind in retained_inds.iter() {

                // check the conditional mutual information for this motif with
                //   each of the retained motifs
                let good_cats = Array::from_vec(self.motifs[*motif_ind].hits.to_vec());
                //println!("Good cats: {:?}", good_cats);
                //println!("Candidate cats: {:?}", cand_cats);
                //println!("y-values: {:?}", y_vals);

                let contingency = info_theory::construct_3d_contingency(
                    cand_cats.view(),
                    y_vals.view(),
                    good_cats.view(),
                );
                //println!("Contingency: {:?}", contingency);
                let cmi = info_theory::conditional_adjusted_mutual_information(
                    contingency.view()
                );
                //println!("CMI: {:?}", cmi);

                // add candidate's parameter number to model
                model_param_num += cand_motif.param_num;
                //println!("model_param_num: {}", model_param_num);
                let log_lik = self.rec_num as f64 * cmi;
                //println!("log_lik: {}", log_lik);
                let this_aic = info_theory::calc_aic(model_param_num, log_lik);
                //println!("this_aic: {}", this_aic);

                // if candidate motif doesn't improve model as added to each of the
                //   chosen motifs, skip it
                if this_aic > 0.0 {
                    // reverse the addition of the parameters for this candidate
                    model_param_num -= cand_motif.param_num;
                    motif_pass = false;
                    break
                }
            }
            if motif_pass {
                retained_inds.push(ind);
                result.push(' ');
                let res = ind.to_string();
                result.push_str(&res);
            }
        }
        result
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let json_fname = &args[1];

    // read in motif info we need
    let file = File::open(json_fname).unwrap();
    // open a buffered reader to open the binary json file
    let buf_reader = BufReader::new(file);
    // hits/amis are sorted by descending ami in python script, no need to sort here
    let motif_data: MotifsData = serde_json::from_reader(buf_reader).unwrap();

    let result = motif_data.get_good_motif_indices();
    if result.is_empty() {
        println!("No motifs are informative");
    }
    println!("{result}");
}

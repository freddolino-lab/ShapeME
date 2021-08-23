use clap;
use motifer;

fn main() {

    // Ideally, all this CLI code should be moved to lib.rs and return a
    //  single struct containing all the args. I was having trouble getting
    //  the infile, params, and out_pref args to work into a struct, though.
    let yaml = clap::load_yaml!("cli.yaml");
    let matches = clap::App::from_yaml(yaml).get_matches();

    let infile = matches.value_of("infile").unwrap().to_string();
    let param_files: Vec<_> = matches.values_of("params").unwrap().collect();
    let param_names: Vec<_> = matches.values_of("param_names").unwrap().collect();
    let kmer: u32 = motifer::fetch_int_arg(matches.value_of("kmer").unwrap());
    let threshold_perc: f32 = motifer::fetch_float_arg(
        matches.value_of("threshold_perc").unwrap()
    );
    let continuous: u32 = motifer::fetch_int_arg(
        matches.value_of("continuous").unwrap()
    );
    let inforobust: u32 = motifer::fetch_int_arg(
        matches.value_of("inforobust").unwrap()
    );
    let frackjack: f32 = motifer::fetch_float_arg(
        matches.value_of("frackjack").unwrap()
    );
    let out_pref = matches.value_of("out_pref").unwrap();
}

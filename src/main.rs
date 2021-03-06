use image;
use std::path::PathBuf;
use structopt::StructOpt;

mod lbp;
mod slic;

use slic::{visualize, Connexity, SLIC};

#[derive(Debug, StructOpt)]
#[structopt(name = "tslic", about = "A Texture Sensitive SLIC implementation.")]
struct Opt {
    #[structopt(short = "m", default_value = "10")]
    m: u32,

    #[structopt(short = "s", default_value = "20")]
    s: u32,

    #[structopt(short = "e", long = "error-threshold", default_value = "50")]
    error_threshold: f32,

    #[structopt(short = "S", default_value = "10")]
    min_size: u32,

    #[structopt(short = "t", default_value = "10")]
    texture: f32,

    #[structopt(long = "c8")]
    c8: bool,

    #[structopt(parse(from_os_str))]
    input: PathBuf,

    /// Output file, stdout if not present
    #[structopt(parse(from_os_str))]
    output: PathBuf,
}

fn main() {
    let opt = Opt::from_args();
    println!("{:?}", opt);

    let img = image::open(opt.input).unwrap();

    let slic = SLIC::new(&img);
    let regions = slic.process(
        opt.m,
        opt.m,
        opt.texture,
        opt.error_threshold,
        opt.min_size,
        if opt.c8 { Connexity::C8 } else { Connexity::C4 },
    );
    let res = visualize(&img, &regions);

    res.save(opt.output).unwrap();
}

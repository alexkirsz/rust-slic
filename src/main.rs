#[macro_use]
extern crate lazy_static;

use glam::{Mat3, Vec2, Vec3};
use image::{self, DynamicImage, GenericImageView, ImageBuffer};
use std::{env, process};

lazy_static! {
    static ref MAT: Mat3 =
        Mat3::from_cols_array(&[0.618, 0.177, 0.205, 0.299, 0.587, 0.114, 0.0, 0.056, 0.944])
            .transpose();
}

fn rgb_to_xyz(x: glam::Vec3) -> glam::Vec3 {
    MAT.mul_vec3(x)
}

lazy_static! {
    static ref XYZ_WHITE: Vec3 = rgb_to_xyz(Vec3::new(1.0, 1.0, 1.0));
}

fn xyz_to_lab(x: glam::Vec3) -> glam::Vec3 {
    fn f(t: f32) -> f32 {
        if t > 0.207 {
            t.powi(3)
        } else {
            (116.0 * t - 16.0) / 903.3
        }
    }

    let ratio = x / *XYZ_WHITE;
    let l = if ratio.y() > 0.008856 {
        116.0 * ratio.y().powf(1.0 / 3.0) - 16.0
    } else {
        903.3 * ratio.y()
    };
    let a = 500.0 * (f(ratio.x()) - f(ratio.y()));
    let b = 200.0 * (f(ratio.y()) - f(ratio.z()));

    Vec3::new(l, a, b)
}

fn norm(x: Vec3, coords: Vec2, ratio: f32) -> f32 {
    x.length() + ratio * coords.length()
}

#[derive(Debug)]
struct Labxy(Vec3, Vec2);

fn sample(pixels: &Vec<Vec3>, width: u32, height: u32, s: u32, n: u32) -> Vec<Vec<Labxy>> {
    let x_steps = (width + 1) / s;
    let y_steps = (height + 1) / s;
    let mut clusters = Vec::with_capacity((x_steps * y_steps) as usize);
    for xx in 0..x_steps {
        let nx = xx * s + s / 2;
        for yy in 0..y_steps {
            let ny = yy * s + s / 2;
            let mut smallest_grad = std::f32::INFINITY;
            let mut smallest_pos: Option<Labxy> = None;
            for x in (nx - n / 2)..=(nx + n / 2) {
                if x >= width {
                    continue;
                }
                for y in (ny - n / 2)..=(ny + n / 2) {
                    if y >= height || x == nx && y == ny {
                        continue;
                    }
                    let grad = (pixels[(y * width + x + 1) as usize]
                        - pixels[(y * width + x - 1) as usize])
                        .length()
                        + (pixels[((y + 1) * width + x) as usize]
                            - pixels[((y - 1) * width + x) as usize])
                            .length();
                    if grad < smallest_grad {
                        smallest_grad = grad;
                        smallest_pos = Some(Labxy(
                            pixels[(y * width + x) as usize],
                            Vec2::new(x as f32, y as f32),
                        ));
                    }
                }
            }
            clusters.push(vec![smallest_pos.unwrap()]);
        }
    }
    clusters
}

fn compute_centers(clusters: &Vec<Vec<Labxy>>) -> Vec<Labxy> {
    clusters
        .iter()
        .map(|pixels| {
            let len = pixels.len() as f32;
            let sum = pixels.into_iter().fold(
                Labxy(Vec3::new(0.0, 0.0, 0.0), Vec2::new(0.0, 0.0)),
                |state, x| Labxy(state.0 + x.0, state.1 + x.1),
            );
            Labxy(sum.0 / len, sum.1 / len)
        })
        .collect()
}

fn iterate(
    pixels: &Vec<Vec3>,
    width: u32,
    height: u32,
    clusters: &Vec<Vec<Labxy>>,
    centers: &Vec<Labxy>,
    m: u32,
    s: u32,
) -> Vec<Vec<Labxy>> {
    let mut result: Vec<(f32, Option<u32>)> =
        pixels.iter().map(|_| (std::f32::INFINITY, None)).collect();

    let ratio = (m as f32) / (s as f32);
    for (i, center) in (0u32..).zip(centers.into_iter()) {
        let (cx, cy) = (center.1.x().round() as i64, center.1.y().round() as i64);
        for y in (cy - s as i64)..(cy + s as i64) {
            if y < 0 || y >= height as i64 {
                continue;
            }
            for x in (cx - s as i64)..(cx + s as i64) {
                if x < 0 || x >= width as i64 {
                    continue;
                }
                let idx = ((y as u32) * width + x as u32) as usize;
                let lab = pixels[idx];
                let xy = Vec2::new(x as f32, y as f32);
                let dist = norm(lab - center.0, xy - center.1, ratio);
                if dist < result[idx].0 {
                    result[idx] = (dist, Some(i));
                }
            }
        }
    }

    let mut next_clusters: Vec<Vec<Labxy>> = clusters.iter().map(|_| vec![]).collect();
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let res = &result[idx as usize];
            if let Some(cluster_idx) = res.1 {
                next_clusters[cluster_idx as usize]
                    .push(Labxy(pixels[idx as usize], Vec2::new(x as f32, y as f32)));
            }
        }
    }

    next_clusters
}

fn l1_vec3(v: Vec3) -> f32 {
    v.x() + v.y() + v.z()
}

fn l1_vec2(v: Vec2) -> f32 {
    v.x() + v.y()
}

fn slic(img: &DynamicImage, m: u32, s: u32, threshold: f32) -> Vec<Vec<Labxy>> {
    let pixels = img.pixels().map(|(_x, _y, v)| {
        Vec3::new(
            v[0] as f32 / 255.0,
            v[1] as f32 / 255.0,
            v[2] as f32 / 255.0,
        )
    });

    let lab_pixels: Vec<Vec3> = pixels.map(rgb_to_xyz).map(xyz_to_lab).collect();

    let mut clusters = sample(&lab_pixels, img.width(), img.height(), s, 3);
    let mut centers = compute_centers(&clusters);
    let mut err = std::f32::INFINITY;
    while err > threshold {
        let next_clusters = iterate(
            &lab_pixels,
            img.width(),
            img.height(),
            &clusters,
            &centers,
            m,
            s,
        );
        let next_centers = compute_centers(&next_clusters);
        err = centers
            .iter()
            .zip(next_centers.iter())
            .fold(0.0, |state, (c1, c2)| {
                state + l1_vec3((c1.0 - c2.0).abs()) + l1_vec2((c1.1 - c2.1).abs())
            });
        clusters = next_clusters;
        centers = next_centers;
    }
    return clusters;
}

fn visualize(img: &DynamicImage, clusters: &Vec<Vec<Labxy>>) -> image::RgbImage {
    let mut res = ImageBuffer::new(img.width(), img.height());
    for cluster in clusters {
        let sum = cluster
            .iter()
            .fold(Vec3::new(0.0, 0.0, 0.0), |state, pixel| {
                let rgb = img.get_pixel(pixel.1.x() as u32, pixel.1.y() as u32);
                state + Vec3::new(rgb[0] as f32, rgb[1] as f32, rgb[2] as f32)
            });
        let avg = sum / (cluster.len() as f32);
        let color = image::Rgb([
            avg.x().round() as u8,
            avg.y().round() as u8,
            avg.z().round() as u8,
        ]);
        for pixel in cluster {
            res.put_pixel(pixel.1.x() as u32, pixel.1.y() as u32, color);
        }
    }
    res
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        process::exit(1);
    }

    let img = image::open(&args[1]).unwrap();

    // let res = ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
    //     let p = lab_pixels[(y * img.width() + x) as usize];
    //     // https://stackoverflow.com/a/19099064/969302
    //     image::Rgb([
    //         (p.x() / 100.0 * 255.0).round() as u8,
    //         ((p.y() + 86.185) / 184.439 * 255.0).round() as u8,
    //         ((p.z() + 107.863) / 202.345 * 255.0).round() as u8,
    //     ])
    // });
    // res.save("lol.png").unwrap();

    let clusters = slic(&img, 10, 40, 50.0);
    let res = visualize(&img, &clusters);

    res.save("lol.png").unwrap();
}

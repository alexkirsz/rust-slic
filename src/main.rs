use glam::{Vec2, Vec3};
use image::{self, DynamicImage, GenericImageView, ImageBuffer};
use palette::{Lab, Srgb};
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use std::{env, process};

fn norm(x: Vec3, coords: Vec2, ratio: f32) -> f32 {
    x.length() + ratio * coords.length()
}

#[derive(Debug)]
struct Labxy(Vec3, Vec2);

fn sample(pixels: &Vec<Vec3>, width: u32, height: u32, s: u32, n: u32) -> Vec<Vec<Labxy>> {
    assert!(n <= s);
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
                    if y >= height {
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

fn slic(img: &DynamicImage, m: u32, s: u32, threshold: f32, min_size: u32) -> Vec<Region> {
    let pixels = img.pixels().map(|(_x, _y, v)| {
        Srgb::new(
            v[0] as f32 / 255.0,
            v[1] as f32 / 255.0,
            v[2] as f32 / 255.0,
        )
    });

    let lab_pixels: Vec<Vec3> = pixels
        .into_iter()
        .map(|p| Lab::from(p))
        .map(|p| Vec3::from(p.into_components()))
        .collect();

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

    let labels = clusters_to_labels(&clusters, img.width(), img.height());

    let (connected_labels, labels_count) =
        extract_connected_components(&labels, img.width(), img.height());

    let regions = labels_to_regions(&connected_labels, labels_count, img.width(), img.height());

    let regions_connex = enforce_connectivity(&regions, min_size);

    regions_connex
}

fn clusters_to_labels(clusters: &Vec<Vec<Labxy>>, width: u32, height: u32) -> Vec<usize> {
    let mut labels = vec![0usize; (width * height) as usize];

    for (i, cluster) in (0usize..).zip(clusters.iter()) {
        for pixel in cluster {
            labels[(pixel.1.y() as u32 * width + pixel.1.x() as u32) as usize] = i;
        }
    }

    labels
}

fn extract_connected_components(
    labels: &Vec<usize>,
    width: u32,
    height: u32,
) -> (Vec<usize>, usize) {
    let mut visited = vec![false; labels.len()];
    let mut labels_out = vec![0usize; labels.len()];
    let mut labels_count = 0usize;
    let mut queue: Vec<(u32, u32)> = vec![];

    for sy in 0..height {
        for sx in 0..width {
            if visited[(sy * width + sx) as usize] {
                continue;
            }

            queue.push((sx, sy));

            while let Some((x, y)) = queue.pop() {
                let idx = (y * width + x) as usize;
                visited[idx] = true;
                let value = labels[idx];
                labels_out[idx] = labels_count;

                let snx = if x > 0 { x - 1 } else { x };
                let enx = if x < width - 1 { x + 1 } else { x };
                let sny = if y > 0 { y - 1 } else { y };
                let eny = if y < height - 1 { y + 1 } else { y };

                for ny in sny..=eny {
                    for nx in snx..=enx {
                        let nidx = (ny * width + nx) as usize;
                        if visited[nidx] {
                            continue;
                        }
                        let nvalue = labels[nidx];
                        if nvalue == value {
                            queue.push((nx, ny));
                        }
                    }
                }
            }

            labels_count += 1;
        }
    }

    (labels_out, labels_count)
}

#[derive(Debug)]
struct Region {
    label: usize,
    pixels: Vec<(u32, u32)>,
    neighbors: HashSet<usize>,
}

fn labels_to_regions(labels: &Vec<usize>, n: usize, width: u32, height: u32) -> Vec<Region> {
    let mut regions: Vec<Region> = (0..n)
        .map(|i| Region {
            label: i,
            pixels: vec![],
            neighbors: HashSet::new(),
        })
        .collect();

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width) + x) as usize;
            let label = labels[idx];
            let region = &mut regions[label];
            region.pixels.push((x, y));

            let snx = if x > 0 { x - 1 } else { x };
            let enx = if x < width - 1 { x + 1 } else { x };
            let sny = if y > 0 { y - 1 } else { y };
            let eny = if y < height - 1 { y + 1 } else { y };

            for ny in sny..=eny {
                for nx in snx..=enx {
                    let nidx = (ny * width + nx) as usize;
                    let nlabel = labels[nidx];
                    if nlabel != label {
                        region.neighbors.insert(nlabel);
                    }
                }
            }
        }
    }

    regions
}

fn enforce_connectivity(regions: &Vec<Region>, min_size: u32) -> Vec<Region> {
    let min_size = min_size as usize;

    let mut regions_out: HashMap<usize, Region> =
        HashMap::from_iter(regions.iter().filter_map(|r| {
            if r.pixels.len() > min_size {
                Some((
                    r.label,
                    Region {
                        label: r.label,
                        pixels: r.pixels.clone(),
                        neighbors: r.neighbors.clone(),
                    },
                ))
            } else {
                None
            }
        }));

    let mut parent_labels: Vec<usize> = regions
        .iter()
        .map(|r| {
            if r.pixels.len() > min_size {
                r.label
            } else {
                let biggest_neighbor = r
                    .neighbors
                    .iter()
                    .map(|nlabel| &regions[*nlabel])
                    .max_by_key(|neighbor| neighbor.pixels.len())
                    .unwrap();
                biggest_neighbor.label
            }
        })
        .collect();

    for label in 0usize..parent_labels.len() {
        let mut parent_label = parent_labels[label];
        if label == parent_label {
            continue;
        }
        while parent_label != parent_labels[parent_label] {
            let next_parent_label = parent_labels[parent_label];
            parent_labels[parent_label] = label;
            parent_label = next_parent_label;
        }
        let region = &regions[label];
        // FIXME: It's still possible to end up in a local maxima that doesn't
        // respect the min_size predicate.
        let parent_region = regions_out.get_mut(&parent_label).unwrap();
        parent_region.pixels.extend(&region.pixels);
        parent_region.neighbors.extend(&region.neighbors);
    }

    regions_out.into_iter().map(|(_, v)| v).collect()
}

fn visualize(img: &DynamicImage, regions: &Vec<Region>) -> image::RgbImage {
    let mut res = ImageBuffer::new(img.width(), img.height());
    for region in regions {
        let sum = region
            .pixels
            .iter()
            .fold(Vec3::new(0.0, 0.0, 0.0), |state, pixel| {
                let rgb = img.get_pixel(pixel.0 as u32, pixel.1 as u32);
                state + Vec3::new(rgb[0] as f32, rgb[1] as f32, rgb[2] as f32)
            });
        let avg = sum / (region.pixels.len() as f32);
        let color = image::Rgb([
            avg.x().round() as u8,
            avg.y().round() as u8,
            avg.z().round() as u8,
        ]);
        for pixel in &region.pixels {
            res.put_pixel(pixel.0, pixel.1, color);
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

    let regions = slic(&img, 10, 20, 50.0, 10);
    let res = visualize(&img, &regions);

    res.save("lol.png").unwrap();
}

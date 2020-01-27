use glam::{Vec2, Vec3};
use image::{self, DynamicImage, GenericImageView, ImageBuffer};
use palette::{Lab, Srgb};
use std::collections::VecDeque;
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use std::ops;

use crate::lbp::lbp;

#[derive(Debug, Copy, Clone)]
struct PointMeta {
    lab_color: Vec3,
    coordinates: Vec2,
    lbp: u8,
}

#[derive(Debug)]
pub struct Region {
    label: usize,
    pixels: Vec<Coord>,
    neighbors: HashSet<usize>,
}

fn l1_vec3(v: Vec3) -> f32 {
    v.x().abs() + v.y().abs() + v.z().abs()
}

fn l1_vec2(v: Vec2) -> f32 {
    v.x().abs() + v.y().abs()
}

impl ops::Sub<PointMeta> for PointMeta {
    type Output = PointMeta;

    fn sub(self, rhs: PointMeta) -> PointMeta {
        PointMeta {
            lab_color: self.lab_color - rhs.lab_color,
            coordinates: self.coordinates - rhs.coordinates,
            lbp: self.lbp ^ rhs.lbp,
        }
    }
}

impl PointMeta {
    fn new(lab_color: Vec3, coordinates: Vec2, lbp: u8) -> PointMeta {
        PointMeta {
            lab_color,
            coordinates,
            lbp,
        }
    }

    fn norm(&self, ratio: f32, texture_coef: f32) -> f32 {
        self.lab_color.length()
            + ratio * self.coordinates.length()
            + (self.lbp.count_ones() as f32).sqrt() * texture_coef
    }

    fn l1_norm(&self, texture_coef: f32) -> f32 {
        l1_vec3(self.lab_color)
            + l1_vec2(self.coordinates)
            + self.lbp.count_ones() as f32 * texture_coef
    }
}

pub struct SLIC {
    lab_pixels: Vec<Vec3>,
    lbp: Vec<u8>,
    width: u32,
    height: u32,
    m: u32,
    s: u32,
    texture_coef: f32,
}

#[derive(Debug, Copy, Clone)]
struct Coord(u32, u32);

impl SLIC {
    fn sample(&self, coord: Coord) -> PointMeta {
        let x = coord.0;
        let y = coord.1;
        let index = (y * self.width + x) as usize;
        PointMeta {
            lab_color: self.lab_pixels[index],
            coordinates: Vec2::new(x as f32, y as f32),
            lbp: self.lbp[index],
        }
    }

    /* pick initial clusters at least gradient coordinates */
    fn sample_initial_centers(&self, n: u32) -> Vec<Vec<Coord>> {
        assert!(n <= self.s);
        let x_steps = (self.width + 1) / self.s;
        let y_steps = (self.height + 1) / self.s;
        let mut clusters = Vec::with_capacity((x_steps * y_steps) as usize);
        for xx in 0..x_steps {
            let nx = xx * self.s + self.s / 2;
            for yy in 0..y_steps {
                let ny = yy * self.s + self.s / 2;
                let mut smallest_grad = std::f32::INFINITY;
                let mut smallest_pos: Option<Coord> = None;
                for x in (nx - n / 2)..=(nx + n / 2) {
                    if x >= self.width {
                        continue;
                    }
                    for y in (ny - n / 2)..=(ny + n / 2) {
                        if y >= self.height {
                            continue;
                        }
                        let grad = (self.lab_pixels[(y * self.width + x + 1) as usize]
                            - self.lab_pixels[(y * self.width + x - 1) as usize])
                            .length()
                            + (self.lab_pixels[((y + 1) * self.width + x) as usize]
                                - self.lab_pixels[((y - 1) * self.width + x) as usize])
                                .length();
                        if grad < smallest_grad {
                            smallest_grad = grad;
                            smallest_pos = Some(Coord(x, y));
                        }
                    }
                }
                clusters.push(vec![smallest_pos.unwrap()]);
            }
        }
        clusters
    }

    fn compute_centers(&self, clusters: &Vec<Vec<Coord>>) -> Vec<PointMeta> {
        clusters
            .iter()
            .map(|pixels| {
                let len = pixels.len();
                let sum = pixels.into_iter().fold(
                    (
                        Vec3::new(0.0, 0.0, 0.0),
                        Vec2::new(0.0, 0.0),
                        vec![0usize; 8],
                    ),
                    |state, x| {
                        let sample = self.sample(*x);
                        let lab_sum = state.0 + sample.lab_color;
                        let coord_sum = state.1 + sample.coordinates;
                        let lbp_sum = (0usize..8)
                            .map(|i| {
                                state.2[i]
                                    + if (sample.lbp & 1 << i) == 0 {
                                        0usize
                                    } else {
                                        1usize
                                    }
                            })
                            .collect();
                        (lab_sum, coord_sum, lbp_sum)
                    },
                );
                let mean_lbp = (0usize..8).zip(sum.2).fold(0u8, |state, (i, count)| {
                    if count > len / 2 {
                        state | 1u8 << i
                    } else {
                        state
                    }
                });
                PointMeta::new(sum.0 / len as f32, sum.1 / len as f32, mean_lbp)
            })
            .collect()
    }

    fn iterate(&self, clusters: &Vec<Vec<Coord>>, centers: &Vec<PointMeta>) -> Vec<Vec<Coord>> {
        let mut result: Vec<(f32, Option<u32>)> =
            vec![(std::f32::INFINITY, None); self.lab_pixels.len()];

        let ratio = (self.m as f32) / (self.s as f32);
        for (i, center) in (0u32..).zip(centers.into_iter()) {
            let center_pos = center.coordinates;
            let (cx, cy) = (center_pos.x().round() as i64, center_pos.y().round() as i64);
            for y in (cy - self.s as i64)..(cy + self.s as i64) {
                if y < 0 || y >= self.height as i64 {
                    continue;
                }
                for x in (cx - self.s as i64)..(cx + self.s as i64) {
                    if x < 0 || x >= self.width as i64 {
                        continue;
                    }

                    let index = y as usize * self.width as usize + x as usize;
                    let dist = (self.sample(Coord(x as u32, y as u32)) - *center)
                        .norm(ratio, self.texture_coef);
                    if dist < result[index].0 {
                        result[index] = (dist, Some(i));
                    }
                }
            }
        }

        let mut next_clusters: Vec<Vec<Coord>> = clusters.iter().map(|_| vec![]).collect();
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y * self.width + x;
                let res = &result[idx as usize];
                if let Some(cluster_idx) = res.1 {
                    next_clusters[cluster_idx as usize].push(Coord(x as u32, y as u32));
                }
            }
        }

        next_clusters
    }

    pub fn new(img: &DynamicImage, m: u32, s: u32, texture_coef: f32) -> SLIC {
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

        SLIC {
            lab_pixels: lab_pixels,
            lbp: lbp(img),
            width: img.width(),
            height: img.height(),
            m: m,
            s: s,
            texture_coef: texture_coef,
        }
    }

    pub fn process(&self, err_threshold: f32, min_size: u32) -> Vec<Region> {
        let mut clusters = self.sample_initial_centers(3);
        println!("initials clusters: {}", clusters.len());
        let mut centers = self.compute_centers(&clusters);
        let mut err = std::f32::INFINITY;
        while err > err_threshold {
            println!("error: {}", err);
            let next_clusters = self.iterate(&clusters, &centers);
            let next_centers = self.compute_centers(&next_clusters);
            err = centers
                .iter()
                .zip(next_centers.iter())
                .fold(0.0, |state, (c1, c2)| {
                    state + (*c1 - *c2).l1_norm(self.texture_coef)
                });
            clusters = next_clusters;
            centers = next_centers;
        }

        let labels = self.clusters_to_labels(&clusters);

        let (connected_labels, labels_count) = self.extract_connected_components(&labels);

        let regions = self.labels_to_regions(&connected_labels, labels_count);

        SLIC::enforce_connectivity(&regions, min_size)
        // regions
    }

    fn clusters_to_labels(&self, clusters: &Vec<Vec<Coord>>) -> Vec<usize> {
        let mut labels = vec![0usize; (self.width * self.height) as usize];

        for (i, cluster) in (0usize..).zip(clusters.iter()) {
            for pixel in cluster {
                labels[(pixel.1 as u32 * self.width + pixel.0 as u32) as usize] = i;
            }
        }

        labels
    }

    fn extract_connected_components(&self, labels: &Vec<usize>) -> (Vec<usize>, usize) {
        let mut visited = vec![false; labels.len()];
        let mut labels_out = vec![0usize; labels.len()];
        let mut labels_count = 0usize;
        let mut queue: Vec<(u32, u32)> = vec![];

        for sy in 0..self.height {
            for sx in 0..self.width {
                if visited[(sy * self.width + sx) as usize] {
                    continue;
                }

                queue.push((sx, sy));

                while let Some((x, y)) = queue.pop() {
                    let idx = (y * self.width + x) as usize;
                    visited[idx] = true;
                    let value = labels[idx];
                    labels_out[idx] = labels_count;

                    let snx = if x > 0 { x - 1 } else { x };
                    let enx = if x < self.width - 1 { x + 1 } else { x };
                    let sny = if y > 0 { y - 1 } else { y };
                    let eny = if y < self.height - 1 { y + 1 } else { y };

                    for ny in sny..=eny {
                        for nx in snx..=enx {
                            let nidx = (ny * self.width + nx) as usize;
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

    fn labels_to_regions(&self, labels: &Vec<usize>, n: usize) -> Vec<Region> {
        let mut regions: Vec<Region> = (0..n)
            .map(|i| Region {
                label: i,
                pixels: vec![],
                neighbors: HashSet::new(),
            })
            .collect();

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = ((y * self.width) + x) as usize;
                let label = labels[idx];
                let region = &mut regions[label];
                region.pixels.push(Coord(x, y));

                let snx = if x > 0 { x - 1 } else { x };
                let enx = if x < self.width - 1 { x + 1 } else { x };
                let sny = if y > 0 { y - 1 } else { y };
                let eny = if y < self.height - 1 { y + 1 } else { y };

                for ny in sny..=eny {
                    for nx in snx..=enx {
                        let nidx = (ny * self.width + nx) as usize;
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
        /*
        here's the plan:

        propagate the reparenting layer by layer.
        first queue the first order children, then the second order children, and on and on.
         */

        let min_size = min_size as usize;

        /* first, partition the regions */
        let (root_regions, orphean_regions): (Vec<&Region>, Vec<&Region>) = regions
            .iter()
            .partition(|region| region.pixels.len() >= min_size);

        /* this map will store the chosen parent for each region */
        let mut region_parents: Vec<Option<usize>> = vec![None; regions.len()];

        /* a bitset of the regions that should currently be queued */
        let mut queued: Vec<bool> = vec![false; regions.len()];
        let mut queue = VecDeque::<(usize, usize)>::with_capacity(regions.len());

        /* setup helper functions */
        let push_region = |queued: &mut Vec<bool>,
                           queue: &mut VecDeque<(usize, usize)>,
                           region_parents: &mut Vec<Option<usize>>,
                           parent: usize,
                           item: usize| {
            if !queued[item] && region_parents[item].is_none() {
                queued[item] = true;
                queue.push_back((parent, item));
            }
        };

        let pop_region = |queued: &mut Vec<bool>, queue: &mut VecDeque<(usize, usize)>| {
            let item = queue.pop_front();
            if !item.is_none() {
                queued[item.unwrap().1] = false;
            }
            item
        };

        /* queue all the non-toplevel neighbors of root regions */
        for root_region in &root_regions {
            region_parents[root_region.label] = Some(root_region.label);
            for root_neighbor in &root_region.neighbors {
                push_region(
                    &mut queued,
                    &mut queue,
                    &mut region_parents,
                    root_region.label,
                    *root_neighbor,
                );
            }
        }

        while let Some((parent_id, orphean_id)) = pop_region(&mut queued, &mut queue) {
            /* set the parent relationship */
            region_parents[orphean_id] = Some(parent_id);

            /* each neighbor has the same parent */
            for neighbor in &regions[orphean_id].neighbors {
                push_region(
                    &mut queued,
                    &mut queue,
                    &mut region_parents,
                    parent_id,
                    *neighbor,
                );
            }
        }

        let mut regions_out: HashMap<usize, Region> =
            HashMap::from_iter(root_regions.iter().filter_map(|region| {
                Some((
                    region.label,
                    Region {
                        label: region.label,
                        pixels: region.pixels.clone(),
                        neighbors: region.neighbors.clone(),
                    },
                ))
            }));

        for orphean_region in orphean_regions {
            let parent_label = region_parents[orphean_region.label].unwrap();
            let parent = regions_out.get_mut(&parent_label).unwrap();
            parent.pixels.extend(&orphean_region.pixels);
            parent.neighbors.extend(&orphean_region.neighbors);
        }

        regions_out.into_iter().map(|(_, v)| v).collect()
    }
} // impl SLIC

pub fn visualize(img: &DynamicImage, regions: &Vec<Region>) -> image::RgbImage {
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

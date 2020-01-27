use image::{DynamicImage, GenericImageView, ImageBuffer};

pub fn lbp(img: &DynamicImage) -> Vec<u8> {
    let (width, height) = (img.width() as usize, img.height() as usize);
    let img = match img.grayscale() {
        image::DynamicImage::ImageLuma8(img) => img,
        _ => panic!("Bad image!"),
    };

    let neighbors: [(i32, i32, u8); 8] = [
        (-1, -1, 1 << 0),
        (0, -1, 1 << 1),
        (1, -1, 1 << 2),
        (1, 0, 1 << 3),
        (1, 1, 1 << 4),
        (0, 1, 1 << 5),
        (-1, 1, 1 << 6),
        (-1, 0, 1 << 7),
    ];

    let mut res = vec![0u8; height * width];
    for (x, y, val) in img.enumerate_pixels() {
        let mut out_val = 0u8;
        for (dx, dy, exp) in &neighbors {
            let (nx, ny) = (x as i32 + *dx, y as i32 + *dy);
            if nx < 0 || nx == (width as i32) || ny < 0 || ny == (height as i32) {
                continue;
            }

            let neighbour = img.get_pixel(nx as u32, ny as u32);
            if neighbour[0] < val[0] {
                out_val |= exp;
            }
        }
        res[y as usize * width + x as usize] = out_val;
    }
    res
}

#[allow(dead_code)]
pub fn visualize_lbp(img: &DynamicImage, ref_x: usize, ref_y: usize) -> image::RgbImage {
    let mut res = ImageBuffer::new(img.width(), img.height());
    let lbp_image = lbp(img);
    let ref_lbp = lbp_image[img.width() as usize * ref_y + ref_x];

    for (x, y, val) in res.enumerate_pixels_mut() {
        let cur_lbp = lbp_image[(img.width() * y + x) as usize];
        let pix_value = ((ref_lbp ^ cur_lbp).count_ones() * 32) as u8;
        let color = image::Rgb([pix_value, pix_value, pix_value]);
        *val = color;
    }

    res
}

fn lbp(img: &DynamicImage) -> Vec<u32> {
    let img = img.grayscale();

    let neighbors: [(i32, i32, u32); 8] = [
        (-1, -1, 1 << 0),
        (0, -1, 1 << 1),
        (1, -1, 1 << 2),
        (1, 0, 1 << 3),
        (1, 1, 1 << 4),
        (0, 1, 1 << 5),
        (-1, 1, 1 << 6),
        (-1, 0, 1 << 7),
    ];

    let res = match img {
        image::DynamicImage::ImageLuma8(img) => {
            let mut res = vec![0u32; (img.height() * img.width()) as usize];
            let (w, h) = (img.width() as i32, img.height() as i32);
            for (x, y, val) in img.enumerate_pixels() {
                let mut out_val = 0u32;
                for (dx, dy, exp) in &neighbors {
                    let (nx, ny) = (x as i32 + *dx, y as i32 + *dy);
                    if nx < 0 || nx == w || ny < 0 || ny == h {
                        continue;
                    }
                    let neighbour = img.get_pixel(nx as u32, ny as u32);
                    let diff = (neighbour[0] as i32) - (val[0] as i32);
                    if diff > 0 {
                        out_val += exp;
                    }
                }
                res[(y * img.width() + x) as usize] = out_val;
            }
            res
        }
        _ => panic!("Bad image!"),
    };

    res
}

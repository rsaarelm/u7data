use std::{
    env,
    fs::File,
    io::{prelude::*, Cursor, SeekFrom},
    path::{Path, PathBuf},
};

use anyhow::{bail, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use clap::Parser;
use image::{ImageBuffer, Rgba};
use itertools::Itertools;

type Pixel = Rgba<u8>;
type Image = ImageBuffer<Pixel, Vec<u8>>;

// Clap CLI interface spec:
//
// Set up an enum with multiple commands:
//   dump: Dump out all the game graphics, with options
//   catalog: Generate a HTML catalogue of game objects, needs IDM object
//     as input.
//
//  There's a global flag u7-path that all the commands use.

#[derive(Debug, Parser)]
struct Args {
    #[command(subcommand)]
    pub cmd: Command,

    /// Path to the game directory.
    #[arg(long)]
    pub u7_path: Option<PathBuf>,
}

#[derive(Debug, Parser)]
enum Command {
    /// Generate a HTML catalogue of game objects.
    Catalog(CatalogArgs),

    /// Dump out all the game graphics.
    Dump(DumpArgs),
}

fn main() -> Result<()> {
    let mut args = Args::parse();

    if args.u7_path.is_none() {
        let Some(path) = env::var_os("U7_PATH") else {
            bail!("Please specify path to the game files in '--u7-path' command-line option or the environment variable 'U7_PATH'");
        };
        args.u7_path = Some(PathBuf::from(path));
    }
    let path = args.u7_path.unwrap();
    let data = U7Data::load(path)?;

    match args.cmd {
        Command::Dump(ref args) => dump(data, args),
        Command::Catalog(ref args) => catalog(data, args),
    }
}

// {{{1 Catalog builder

#[derive(Debug, Parser)]
struct CatalogArgs {
    /// Location of the database describing all physical game objects. Use "-"
    /// to read from stdin.
    #[arg(default_value = "-")]
    pub object_database: PathBuf,
}

fn catalog(_data: U7Data, _args: &CatalogArgs) -> Result<()> {
    todo!();
}

// {{{1 Shape dumper

#[derive(Debug, Parser)]
struct DumpArgs {
    /// Path to the game directory.
    #[arg(long)]
    pub u7_path: Option<PathBuf>,

    /// Whether to add skewed versions of sprites to make textures of.
    #[arg(long)]
    pub include_skewed: bool,

    /// Whether to add base rectangle to sprites.
    #[arg(long)]
    pub add_base: bool,

    /// Keep the original palette's transparent color.
    #[arg(long)]
    pub keep_transparent_color: bool,
}

fn dump(mut data: U7Data, args: &DumpArgs) -> Result<()> {
    // The orange transparent color hurts my eyes, change it to something
    // nicer.
    if !args.keep_transparent_color {
        data.palette[255] = Rgba([0, 255, 255, 0]);
    }

    // Output palette as PNG.
    const PALETTE_SCALE: u32 = 4;
    let palette_img = ImageBuffer::from_fn(16 * PALETTE_SCALE, 16 * PALETTE_SCALE, |x, y| {
        let idx = (y / PALETTE_SCALE) * 16 + (x / PALETTE_SCALE);
        data.palette[idx as usize]
    });
    save_indexed(&palette_img, &data.palette, "palette.png")?;

    // Decorate shapes with geometry info and save as sprite sheets.
    for (i, ss) in data.shapes.iter().enumerate() {
        let filename = format!("{:04}-{}.png", i, data.shape_name(i));
        let mut shapes = Vec::new();

        for s in ss {
            if s.is_empty() && !args.add_base {
                continue;
            }
            let mut shape = s.clone();
            if args.add_base {
                shape.decorate(data.sprite_dims[i], &data.palette);
            }

            shapes.push(shape);

            if args.include_skewed {
                for s in s.make_skewed() {
                    shapes.push(s);
                }
            }
        }

        if shapes.is_empty() {
            eprintln!("Skipping empty shape {filename}");
            continue;
        }
        let sheet = build_sheet(&shapes, &data.palette);
        save_indexed(&sheet, &data.palette, filename)?;
    }
    Ok(())
}

// {{{1 General game data

#[derive(Debug)]
enum Game {
    BlackGate,
    SerpentIsle,
}

impl Game {
    fn detect(path: impl AsRef<Path>) -> Option<Game> {
        let path = path.as_ref();
        if path.join("SERPENT.COM").exists() {
            Some(Game::SerpentIsle)
        } else if path.join("ULTIMA7.COM").exists() {
            Some(Game::BlackGate)
        } else {
            None
        }
    }

    fn num_shapes(&self) -> usize {
        match self {
            Game::BlackGate => 1024,
            Game::SerpentIsle => 1036,
        }
    }

    /// Supply names for shapes that have empty strings in the game string
    /// data.
    fn missing_name(&self, idx: usize) -> Option<&'static str> {
        match self {
            Game::BlackGate => match idx {
                2 => Some("water"),
                19 => Some("water"),
                20 => Some("water"),
                26 => Some("water"),
                30 => Some("water"),
                64 => Some("water"),
                65 => Some("water"),
                83 => Some("grassy_mud"),
                110 => Some("muddy_rock"),
                161 => Some("thatch_roof"),
                162 => Some("thatch_roof"),
                397 => Some("death_bolt"),
                398 => Some("sparkles"),
                676 => Some("fire_bolt"),
                731 => Some("lightning"),
                793 => Some("chasm"),
                874 => Some("chasm"),
                _ => None,
            },
            Game::SerpentIsle => match idx {
                2 => Some("water"),
                19 => Some("water"),
                20 => Some("water"),
                26 => Some("water"),
                30 => Some("water"),
                64 => Some("water"),
                65 => Some("water"),
                75 => Some("chasm"),
                76 => Some("floor"),
                77 => Some("chasm"),
                82 => Some("a_long_fall"),
                83 => Some("a_long_fall"),
                84 => Some("fountain"),
                110 => Some("fountain"),
                112 => Some("muddy_bank"),
                209 => Some("widget"),
                224 => Some("scorch_mark"),
                334 => Some("carpet"),
                397 => Some("fire_spell"),
                398 => Some("sparkles"),
                479 => Some("claw"),
                676 => Some("fire_bolt"),
                731 => Some("lightning"),
                893 => Some("man"),
                945 => Some("automaton"),
                // Extra Avatar sprite sheets in Serpent Isle
                1024..1036 => Some("avatar"),
                _ => None,
            },
        }
    }
}

// {{{1 Graphics and object data structure reading

#[derive(Debug)]
struct U7Data {
    pub game: Game,
    pub shapes: Vec<Vec<Shape>>,
    pub strings: Vec<String>,
    pub palette: Vec<Pixel>,
    pub sprite_dims: Vec<[i32; 3]>,
}

impl U7Data {
    fn load(path: impl AsRef<Path>) -> Result<U7Data> {
        let path = path.as_ref();
        let Some(game) = Game::detect(path) else {
            bail!("Could not find Ultima 7 or Serpent Isle in {path:?}");
        };

        // Extract dimensions in multiples of 8 pixels for the shapes. This
        // includes data for both tiles and shapes.
        let mut sprite_dims = Vec::new();
        {
            let mut reader = File::open(path.join("STATIC/TFA.DAT"))?;
            // 24 bit flag sets. See the data structure reference and parse
            // more of the flags if needed.
            while let (Ok(f1), Ok(_f2), Ok(f3)) =
                (reader.read_u8(), reader.read_u8(), reader.read_u8())
            {
                let z = (f1 >> 5) & 0x7;
                let x = (f3 & 0x7) + 1;
                let y = ((f3 >> 3) & 0x7) + 1;
                sprite_dims.push([x as i32, y as i32, z as i32]);
            }
        }

        // Load game strings.
        let strings = load_flx(path.join("STATIC/TEXT.FLX"))?;
        let strings = strings
            .iter()
            .map(|s| {
                s.iter()
                    .take_while(|&&b| b != 0)
                    .map(|&b| b as char)
                    .collect::<String>()
            })
            .collect::<Vec<_>>();

        // Palette data has components in 0..64 range, needs to be converted
        // to 0..256.
        let palettes = load_flx(path.join("STATIC/PALETTES.FLX"))?;
        let palette: Vec<Pixel> = palettes[0]
            .iter()
            .tuples()
            .map(|(&r, &g, &b)| {
                if r < 64 && g < 64 && b < 64 {
                    Rgba([r * 4, g * 4, b * 4, 255])
                } else {
                    // Weird transparent colors?
                    Rgba([r, g, b, 255])
                }
            })
            .collect();

        let mut shapes = Vec::new();
        for elt in load_flx(path.join("STATIC/SHAPES.VGA"))?
            .iter()
            .take(game.num_shapes())
        {
            let frames = load_shapes(&mut Cursor::new(elt), &palette)?;
            shapes.push(frames);
        }

        Ok(U7Data {
            game,
            shapes,
            strings,
            palette,
            sprite_dims,
        })
    }

    pub fn shape_name(&self, idx: usize) -> String {
        if idx < self.strings.len() {
            if self.strings[idx].is_empty() || idx >= 1024 {
                if let Some(name) = self.game.missing_name(idx) {
                    return name.to_string();
                }
            } else {
                // Sanitize.
                return self.strings[idx]
                    .chars()
                    .map(|c| {
                        if c.is_alphanumeric() {
                            c.to_ascii_lowercase()
                        } else {
                            '_'
                        }
                    })
                    .collect();
            }
        }

        "unknown".to_owned()
    }
}

/// Load contents from a FLX file.
pub fn load_flx(path: impl AsRef<Path>) -> Result<Vec<Vec<u8>>> {
    let mut reader = File::open(path.as_ref())?;

    // Skip text comment
    reader.seek(SeekFrom::Current(80))?;

    // Magic number
    let magic = reader.read_u32::<LittleEndian>()?;
    if magic != 0xffff1a00 {
        return Err(anyhow::anyhow!("Invalid magic number"));
    }

    let count = reader.read_u32::<LittleEndian>()? as usize;

    // Header junk.
    reader.seek(SeekFrom::Current(40))?;

    let mut records = Vec::new();

    for _ in 0..count {
        let offset = reader.read_u32::<LittleEndian>()? as u64;
        let length = reader.read_u32::<LittleEndian>()? as usize;
        records.push((offset, length));
    }

    let mut ret = Vec::new();

    for (offset, length) in records {
        reader.seek(SeekFrom::Start(offset))?;
        let mut data = vec![0; length];
        reader.read_exact(&mut data)?;
        ret.push(data);
    }

    Ok(ret)
}

// {{{1 Shape type

#[derive(Clone, Debug)]
pub enum Shape {
    Sprite { image: Image, offset: [i32; 2] },

    // Always 8x8.
    Tile(Image),
}

impl AsRef<Image> for Shape {
    fn as_ref(&self) -> &Image {
        match self {
            Shape::Sprite { image, .. } => image,
            Shape::Tile(image) => image,
        }
    }
}

impl Shape {
    pub fn is_empty(&self) -> bool {
        self.as_ref()
            .enumerate_pixels()
            .all(|(_, _, p)| p.0[3] == 0)
    }

    /// Draw a bounding box behind the shape based on the dimensions.
    pub fn decorate(&mut self, dim: [i32; 3], palette: &[Pixel]) {
        let is_empty = self.is_empty();

        let Shape::Sprite {
            ref mut image,
            offset,
        } = self
        else {
            return;
        };

        let mut plot = |force: bool, x: i32, y: i32, color| {
            let x = x - offset[0];
            let y = y - offset[1];
            if x >= 0 && x < image.width() as i32 && y >= 0 && y < image.height() as i32 {
                let p = image.get_pixel(x as u32, y as u32);
                if force || p.0[3] == 0 {
                    image.put_pixel(x as u32, y as u32, color);
                }
            }
        };

        // Mark empty frames with a blank box.
        if is_empty {
            // Color 123 should be a gray pixel.
            *image = ImageBuffer::from_pixel(8, 8, palette[123]);
            return;
        }

        // Solid-color base for footprint.
        for y in 0..dim[1] * 8 {
            for x in 0..dim[0] * 8 {
                let force = (x == 0 && y == 0) || (x == dim[0] * 8 - 1 && y == dim[1] * 8 - 1);
                plot(force, -x, -y, palette[254]);
            }
        }
    }

    pub fn make_skewed(&self) -> Vec<Shape> {
        let mut ret = Vec::new();

        // Don't skew ground tiles.
        let Shape::Sprite { ref image, .. } = self else {
            return ret;
        };

        let south_face = Image::from_fn(image.width() + image.height(), image.height(), |x, y| {
            let (x, y) = (x as i32, y as i32);
            let (w, h) = (image.width() as i32, image.height() as i32);
            let x = x + y - h;
            if x >= 0 && x < w && y >= 0 && y < h {
                *image.get_pixel(x as u32, y as u32)
            } else {
                Rgba([0, 0, 0, 0])
            }
        });

        let east_face = Image::from_fn(image.width(), image.height() + image.width(), |x, y| {
            let (x, y) = (x as i32, y as i32);
            let (w, h) = (image.width() as i32, image.height() as i32);
            let y = y + x - w;
            if x >= 0 && x < w && y >= 0 && y < h {
                *image.get_pixel(x as u32, y as u32)
            } else {
                Rgba([0, 0, 0, 0])
            }
        });

        ret.push(Shape::Sprite {
            image: south_face,
            offset: [0, 0],
        });
        ret.push(Shape::Sprite {
            image: east_face,
            offset: [0, 0],
        });

        ret
    }
}

pub fn load_shapes<R: Read + Seek>(reader: &mut R, palette: &[Pixel]) -> Result<Vec<Shape>> {
    let size = reader.read_u32::<LittleEndian>()? as usize;
    let input_len = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(4))?;

    let mut ret = Vec::new();

    // Shape block does not start with an accurate size value, assume it's a
    // bunch of 8x8 tiles.
    if size != input_len as usize {
        reader.seek(SeekFrom::Start(0))?;

        for _ in 0..(input_len / 64) {
            ret.push(load_tile(reader, palette)?);
        }

        return Ok(ret);
    }

    let mut offsets = Vec::new();
    offsets.push(reader.read_u32::<LittleEndian>()? as u64);
    for _ in 1..(offsets[0] - 4) / 4 {
        offsets.push(reader.read_u32::<LittleEndian>()? as u64);
    }

    for offset in offsets {
        reader.seek(SeekFrom::Start(offset))?;
        ret.push(load_sprite(reader, palette)?);
    }

    Ok(ret)
}

fn load_tile<R: Read + Seek>(reader: &mut R, palette: &[Pixel]) -> Result<Shape> {
    let mut image = ImageBuffer::new(8, 8);

    for y in 0..8 {
        for x in 0..8 {
            image.put_pixel(x, y, palette[reader.read_u8()? as usize]);
        }
    }

    Ok(Shape::Tile(image))
}

fn load_sprite<R: Read + Seek>(reader: &mut R, palette: &[Pixel]) -> Result<Shape> {
    let max_x = reader.read_u16::<LittleEndian>()? as i16 + 1;
    let min_x = -(reader.read_u16::<LittleEndian>()? as i16);
    let min_y = -(reader.read_u16::<LittleEndian>()? as i16);
    let max_y = reader.read_u16::<LittleEndian>()? as i16 + 1;

    let mut image = ImageBuffer::new((max_x - min_x) as u32, (max_y - min_y) as u32);

    let mut plot = |x, y, color| {
        image.put_pixel(
            (x - min_x) as u32,
            (y - min_y) as u32,
            palette[color as usize],
        );
    };

    loop {
        let data = reader.read_u16::<LittleEndian>()?;
        if data == 0 {
            break;
        }

        let x = reader.read_i16::<LittleEndian>()?;
        let y = reader.read_i16::<LittleEndian>()?;

        assert!(x >= min_x && x < max_x && y >= min_y && y < max_y);

        let len = (data >> 1) as usize;
        let mode = data & 1;

        if mode == 0 {
            // Raw data
            for i in 0..len {
                plot(x + i as i16, y, reader.read_u8()?);
            }
        } else {
            // RLE data
            let mut x2 = x;
            while x2 < x + len as i16 {
                let run = reader.read_u8()?;
                let run_type = run & 1;
                let run_len = (run >> 1) as usize;
                if run_type == 0 {
                    // Raw data.
                    for _ in 0..run_len {
                        plot(x2, y, reader.read_u8()?);
                        x2 += 1;
                    }
                } else {
                    // Repeat value
                    let value = reader.read_u8()?;
                    for _ in 0..run_len {
                        plot(x2, y, value);
                        x2 += 1;
                    }
                }
            }
        }
    }

    Ok(Shape::Sprite {
        image,
        offset: [min_x as i32, min_y as i32],
    })
}

// {{{1 utilities

fn build_sheet(shapes: &[Shape], palette: &[Pixel]) -> Image {
    const COLUMNS: usize = 9;

    // Determine positions of images and total sheet dimensions.
    let mut width = 0;
    let mut height = 0;
    let mut positions = Vec::new();
    let mut x = 0;
    let mut y = 0;
    for (i, s) in shapes.iter().enumerate() {
        if i % COLUMNS == 0 {
            x = 0;
            y = height;
        }

        let (mut w, mut h) = s.as_ref().dimensions();

        if matches!(s, Shape::Sprite { .. }) {
            // It's a sprite, add padding.
            positions.push((x + 1, y + 1));
            w += 1;
            h += 1;
        } else {
            // Pack tiles tightly.
            positions.push((x, y));
        }

        width = width.max(x + w);
        height = height.max(y + h);
        x += w;
    }

    let mut result = ImageBuffer::from_pixel(width, height, palette[255]);

    for (s, (x, y)) in shapes.iter().zip(positions) {
        for (sx, sy, p) in s.as_ref().enumerate_pixels() {
            // Skip transparent pixels.
            if p.0[3] != 0 {
                result.put_pixel(x + sx, y + sy, *p);
            }
        }
    }

    result
}

fn save_indexed(image: &Image, palette: &[Pixel], path: impl AsRef<Path>) -> Result<()> {
    let image_data: Vec<u8> = image
        .pixels()
        .map(|p| palette.iter().position(|c| c == p).unwrap_or(0) as u8)
        .collect();

    let palette_data: Vec<u8> = palette
        .iter()
        .flat_map(|p| [p.0[0], p.0[1], p.0[2]])
        .collect();

    let mut writer = File::create(path)?;
    let mut encoder = png::Encoder::new(&mut writer, image.width(), image.height());
    encoder.set_color(png::ColorType::Indexed);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_palette(&palette_data);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&image_data)?;
    Ok(())
}

// vim:foldmethod=marker

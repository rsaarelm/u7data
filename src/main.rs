use std::{
    collections::BTreeMap,
    env,
    fmt::{self, Write},
    fs::File,
    io::{prelude::*, Cursor, SeekFrom},
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::{bail, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use clap::Parser;
use glam::{ivec2, ivec3, IVec2, IVec3};
use image::{ImageBuffer, Rgba};
use itertools::Itertools;
use serde::Deserialize;
use serde_with::DeserializeFromStr;

type Pixel = Rgba<u8>;
type Image = ImageBuffer<Pixel, Vec<u8>>;

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

fn catalog(mut data: U7Data, args: &CatalogArgs) -> Result<()> {
    use Facing::*;

    // Nice transparent color.
    data.palette[255] = Rgba([0, 255, 255, 0]);

    // Read object database from args.
    let db: String = if args.object_database == PathBuf::from("-") {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        buf
    } else {
        std::fs::read_to_string(&args.object_database)?
    };

    let db: BTreeMap<Natural, BTreeMap<ViewKey, View>> = idm::from_str(&db)?;

    // Crummiest possible initial catalog...
    let mut html = String::new();
    // XXX: Askama templates would be nicer than spewing out raw HTML.
    writeln!(
        html,
        "<html><head><title>Ultima 7 Object Catalog</title></head><body><ul>"
    )?;

    for (name, views) in db {
        // The Natural wrapper has served is purpose controlling the BTreeMap
        // order.
        let name = name.0;

        // Filter out empty frames from views
        let views = views
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    View {
                        frames: v
                            .frames
                            .into_iter()
                            .filter(|f| {
                                data.shapes[f.shape].len() > f.frame
                                    && !data.shapes[f.shape][f.frame].is_empty()
                            })
                            .collect(),
                        ..v
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();

        // Object is built from multiple shapes and we need to reassemble it.
        let is_composite = views.iter().any(|(k, _)| k.extension != IVec3::ZERO);

        let mut north_frames = Vec::new();
        let mut south_frames = Vec::new();

        if is_composite {
            for face in [North, West, South, East] {
                let mut frames = Vec::new();
                for (k, v) in &views {
                    if k.facing == face && !v.frames.is_empty() {
                        // XXX: I'll be ignoring animations with composite
                        // objects and just drawing the first frame for now.
                        let f = v.frames[0];
                        let mut f = data.shapes[f.shape][f.frame].clone();
                        f.offset -= v.offset;
                        frames.push((k.extension, f));
                    }
                }

                if !frames.is_empty() {
                    let mut frame = Frame::compose(&data.palette, frames.into_iter());
                    match face {
                        North => north_frames.push(frame),
                        West => {
                            frame.rotate();
                            north_frames.push(frame);
                        }
                        South => south_frames.push(frame),
                        East => {
                            frame.rotate();
                            south_frames.push(frame);
                        }
                    }
                }
            }
        } else {
            for (k, v) in &views {
                for f in &v.frames {
                    let mut frame = data.shapes[f.shape][f.frame].clone();
                    frame.offset -= v.offset;
                    match k.facing {
                        North => north_frames.push(frame),
                        West => {
                            frame.rotate();
                            north_frames.push(frame);
                        }
                        South => south_frames.push(frame),
                        East => {
                            frame.rotate();
                            south_frames.push(frame);
                        }
                    }
                }
            }
        }

        if north_frames.is_empty() && south_frames.is_empty() {
            bail!("No views for object {name}");
        }

        let main_shape = if south_frames.is_empty() {
            &north_frames[0]
        } else {
            &south_frames[0]
        };

        // Save shape as "{name}.png"
        let filename = format!("{name}.png");
        save_indexed(&main_shape.image, &data.palette, &filename)?;

        writeln!(
            html,
            "<li><img src='{filename}' alt='{name}'/> {name} </li>",
        )?;
    }

    writeln!(html, "</ul/></body></html>")?;

    // Write html to "index.html".
    std::fs::write("index.html", html)?;

    Ok(())
}

#[derive(Clone, Debug, Deserialize)]
struct View {
    #[serde(default)]
    pub offset: IVec2,
    pub frames: Vec<FrameIdx>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, DeserializeFromStr)]
enum Facing {
    North,
    East,
    South,
    West,
}

impl FromStr for Facing {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Facing> {
        match s {
            "north" => Ok(Facing::North),
            "east" => Ok(Facing::East),
            "south" => Ok(Facing::South),
            "west" => Ok(Facing::West),
            _ => Err(anyhow::anyhow!("invalid facing")),
        }
    }
}

impl fmt::Display for Facing {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Facing::North => write!(f, "north"),
            Facing::East => write!(f, "east"),
            Facing::South => write!(f, "south"),
            Facing::West => write!(f, "west"),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, DeserializeFromStr)]
struct ViewKey {
    pub facing: Facing,
    pub extension: IVec3,
}

impl From<Facing> for ViewKey {
    fn from(facing: Facing) -> Self {
        ViewKey {
            facing,
            extension: IVec3::ZERO,
        }
    }
}

impl Ord for ViewKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let a = (self.facing, self.extension.to_array());
        let b = (other.facing, other.extension.to_array());
        a.cmp(&b)
    }
}

impl PartialOrd for ViewKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl FromStr for ViewKey {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<ViewKey> {
        let parts = s.split_whitespace().collect::<Vec<_>>();
        match parts.len() {
            // Just a name, assume extension is zero.
            1 => Ok(ViewKey {
                facing: parts[0].parse()?,
                extension: IVec3::ZERO,
            }),
            // Name and x y extension.
            3 => Ok(ViewKey {
                facing: parts[0].parse()?,
                extension: ivec3(parts[1].parse()?, parts[2].parse()?, 0),
            }),
            // Name and x y z extension.
            4 => Ok(ViewKey {
                facing: parts[0].parse()?,
                extension: ivec3(parts[1].parse()?, parts[2].parse()?, parts[3].parse()?),
            }),
            _ => Err(anyhow::anyhow!("invalid view key format")),
        }
    }
}

#[derive(Copy, Clone, Debug, DeserializeFromStr)]
struct FrameIdx {
    pub shape: usize,
    pub frame: usize,
}

impl FromStr for FrameIdx {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<FrameIdx> {
        // Turn "312:16" into FrameIdx { shape: 312, frame: 16 }
        let parts = s.split(':').collect::<Vec<_>>();
        match parts[..] {
            [a, b] => Ok(FrameIdx {
                shape: a.parse()?,
                frame: b.parse()?,
            }),
            _ => Err(anyhow::anyhow!("invalid frame index format {s:?}")),
        }
    }
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
                shape.decorate(&data.palette);
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
    pub shapes: Vec<Vec<Frame>>,
    pub strings: Vec<String>,
    pub palette: Vec<Pixel>,
}

impl U7Data {
    fn load(path: impl AsRef<Path>) -> Result<U7Data> {
        let path = path.as_ref();
        let Some(game) = Game::detect(path) else {
            bail!("Could not find Ultima 7 or Serpent Isle in {path:?}");
        };

        // Extract dimensions in multiples of 8 pixels for the shapes. This
        // includes data for both tiles and shapes.
        let mut shape_dims = Vec::new();
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
                shape_dims.push(ivec3(x as i32, y as i32, z as i32));
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
        for (i, elt) in load_flx(path.join("STATIC/SHAPES.VGA"))?
            .iter()
            .enumerate()
            .take(game.num_shapes())
        {
            let frames = load_frames(&mut Cursor::new(elt), &palette, shape_dims[i])?;
            shapes.push(frames);
        }

        Ok(U7Data {
            game,
            shapes,
            strings,
            palette,
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

// {{{1 Frame type

#[derive(Clone, Default)]
pub struct Frame {
    image: Image,
    // Coordinates of bottom right corner of base in image.
    offset: IVec2,
    // 3D dimensions in multiples of 8 pixels.
    dim: IVec3,
}

impl Frame {
    /// Load a 8x8 tile frame from the Ultima 7 graphics data.
    pub fn tile<R: Read + Seek>(reader: &mut R, palette: &[Pixel]) -> Result<Self> {
        let mut image = ImageBuffer::new(8, 8);

        for y in 0..8 {
            for x in 0..8 {
                image.put_pixel(x, y, palette[reader.read_u8()? as usize]);
            }
        }

        Ok(Frame {
            image,
            offset: IVec2::ZERO,
            dim: ivec3(1, 1, 0),
        })
    }

    /// Load a sprite frame from the Ultima 7 graphics data.
    pub fn sprite<R: Read + Seek>(reader: &mut R, palette: &[Pixel], dim: IVec3) -> Result<Self> {
        let max_x = reader.read_u16::<LittleEndian>()? as i16 + 1;
        let min_x = -(reader.read_u16::<LittleEndian>()? as i16);
        let min_y = -(reader.read_u16::<LittleEndian>()? as i16);
        let max_y = reader.read_u16::<LittleEndian>()? as i16 + 1;

        let mut image = ImageBuffer::from_pixel(
            (max_x - min_x) as u32,
            (max_y - min_y) as u32,
            Rgba([0, 255, 255, 0]),
        );

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

        Ok(Frame {
            image,
            offset: ivec2(min_x as i32, min_y as i32),
            dim,
        })
    }

    /// Construct a composite frame out of multiple sub-frames. The input
    /// contains 3D offsets in multiples of 8 pixels (z axis maps to (-4, -4)
    /// in x, y). The order of inputs is unspecified, so the function must
    /// collect the input first and then determine the correct draw order.
    pub fn compose(palette: &[Pixel], input: impl Iterator<Item = (IVec3, Frame)>) -> Self {
        // TODO: Figure out if all this stuff actually works correctly...
        let mut frames: Vec<(IVec3, Frame)> = input.collect();

        if frames.is_empty() {
            return Default::default();
        }

        // Project upper corner of the object on the view screen's normal
        // vector, sort by that. Hope that's good enough.
        frames.sort_by_key(|(t, frame)| {
            // Offset point is at the lower right corner of the bounding box
            // so x and y don't need to be adjusted by dim.
            let (x, y, z) = (t.x, t.y, t.z + frame.dim[2]);
            x + y + z
        });

        // Establish composite image dimensions.
        let mut min = IVec2::splat(i32::MAX);
        let mut max = IVec2::splat(i32::MIN);

        // Combined tile space bounding box.
        let mut t_min = IVec3::splat(i32::MAX);
        let mut t_max = IVec3::splat(i32::MAX);

        let mut new_offset = IVec2::splat(i32::MIN);

        for (pos, frame) in &frames {
            let (a, b) = frame.corners();
            min = min.min(a + pos.t2s());
            max = max.max(b + pos.t2s());

            new_offset = new_offset.max(pos.t2s());

            t_min = t_min.min(*pos);
            t_max = t_max.max(*pos + frame.dim);
        }

        let mut canvas =
            ImageBuffer::from_pixel((max.x - min.x) as u32, (max.y - min.y) as u32, palette[255]);

        for (pos, frame) in frames {
            frame.draw(&mut canvas, pos.t2s() - min);
        }

        Frame {
            image: canvas,
            offset: new_offset,
            dim: t_max - t_min,
        }
    }

    /// Return true if the frame has no visible pixels.
    pub fn is_empty(&self) -> bool {
        self.image.enumerate_pixels().all(|(_, _, p)| p.0[3] == 0)
    }

    /// Get upper left and lower right corners of the frame.
    pub fn corners(&self) -> (IVec2, IVec2) {
        let min = self.offset;
        let max = min + ivec2(self.image.width() as i32, self.image.height() as i32);
        (min, max)
    }

    /// Draw this frame on another image.
    pub fn draw(&self, canvas: &mut Image, pos: IVec2) {
        for (x, y, p) in self.image.enumerate_pixels() {
            let x = x as i32 + pos.x + self.offset.x;
            let y = y as i32 + pos.y + self.offset.y;
            if p.0[3] != 0
                && x >= 0
                && x < canvas.width() as i32
                && y >= 0
                && y < canvas.height() as i32
            {
                canvas.put_pixel(x as u32, y as u32, *p);
            }
        }
    }

    /// Draw a bounding box behind the frame based on the dimensions.
    pub fn decorate(&mut self, palette: &[Pixel]) {
        let is_empty = self.is_empty();

        let mut plot = |force: bool, x: i32, y: i32, color| {
            let x = x - self.offset[0];
            let y = y - self.offset[1];
            if x >= 0 && x < self.image.width() as i32 && y >= 0 && y < self.image.height() as i32 {
                let p = self.image.get_pixel(x as u32, y as u32);
                if force || p.0[3] == 0 {
                    self.image.put_pixel(x as u32, y as u32, color);
                }
            }
        };

        // Mark empty frames with a blank box.
        if is_empty {
            // Color 123 should be a gray pixel.
            self.image = ImageBuffer::from_pixel(8, 8, palette[123]);
            return;
        }

        // Solid-color base for footprint.
        for y in 0..self.dim.y * 8 {
            for x in 0..self.dim.x * 8 {
                let force =
                    (x == 0 && y == 0) || (x == self.dim.x * 8 - 1 && y == self.dim.y * 8 - 1);
                plot(force, -x, -y, palette[254]);
            }
        }
    }

    pub fn make_skewed(&self) -> Vec<Frame> {
        let mut ret = Vec::new();

        // Don't skew flat shapes.
        if self.dim.z == 0 {
            return ret;
        }

        let south_face = Image::from_fn(
            self.image.width() + self.image.height(),
            self.image.height(),
            |x, y| {
                let (x, y) = (x as i32, y as i32);
                let (w, h) = (self.image.width() as i32, self.image.height() as i32);
                let x = x + y - h;
                if x >= 0 && x < w && y >= 0 && y < h {
                    *self.image.get_pixel(x as u32, y as u32)
                } else {
                    Rgba([0, 0, 0, 0])
                }
            },
        );

        let east_face = Image::from_fn(
            self.image.width(),
            self.image.height() + self.image.width(),
            |x, y| {
                let (x, y) = (x as i32, y as i32);
                let (w, h) = (self.image.width() as i32, self.image.height() as i32);
                let y = y + x - w;
                if x >= 0 && x < w && y >= 0 && y < h {
                    *self.image.get_pixel(x as u32, y as u32)
                } else {
                    Rgba([0, 0, 0, 0])
                }
            },
        );

        ret.push(Frame {
            image: south_face,
            ..Default::default()
        });
        ret.push(Frame {
            image: east_face,
            ..Default::default()
        });

        ret
    }

    /// Turn an east view into a south view or a west view into a north view.
    pub fn rotate(&mut self) {
        let mut image = ImageBuffer::new(self.image.height(), self.image.width());
        for (x, y, p) in self.image.enumerate_pixels() {
            image.put_pixel(y, x, *p);
        }

        self.image = image;
        self.offset = ivec2(self.offset.y, self.offset.x);
        self.dim = ivec3(self.dim.y, self.dim.x, self.dim.z);
    }
}

impl fmt::Debug for Frame {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Frame {{ image_size: {:?}, dim: {:?}, offset: {:?} }}",
            self.image.dimensions(),
            self.dim,
            self.offset
        )
    }
}

pub fn load_frames<R: Read + Seek>(
    reader: &mut R,
    palette: &[Pixel],
    dim: IVec3,
) -> Result<Vec<Frame>> {
    let size = reader.read_u32::<LittleEndian>()? as usize;
    let input_len = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(4))?;

    let mut ret = Vec::new();

    // Shape block does not start with an accurate size value, assume it's a
    // bunch of 8x8 tiles.
    if size != input_len as usize {
        reader.seek(SeekFrom::Start(0))?;

        for _ in 0..(input_len / 64) {
            ret.push(Frame::tile(reader, palette)?);
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
        ret.push(Frame::sprite(reader, palette, dim)?);
    }

    Ok(ret)
}

// {{{1 utilities

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

fn build_sheet(shapes: &[Frame], palette: &[Pixel]) -> Image {
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

        let (mut w, mut h) = s.image.dimensions();

        positions.push((x + 1, y + 1));
        w += 1;
        h += 1;

        width = width.max(x + w);
        height = height.max(y + h);
        x += w;
    }

    let mut result = ImageBuffer::from_pixel(width, height, palette[255]);

    for (s, (x, y)) in shapes.iter().zip(positions) {
        for (sx, sy, p) in s.image.enumerate_pixels() {
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

fn make_gif_animation(frames: &[Frame]) -> Result<Vec<u8>> {
    // Figure out the bounds of the animation frame, the max bounds from all
    // frames.
    //
    // All frames must by drawn centered on their offset.

    let mut min = IVec2::splat(i32::MAX);
    let mut max = IVec2::splat(i32::MIN);
    for f in frames {
        let (a, b) = f.corners();
        min = min.min(a);
        max = max.max(b);
    }

    todo!()
}

pub trait TileSpace: Into<[i32; 3]> + Copy {
    /// Project 3D tile space coordinates into 2D screen space.
    fn t2s(self) -> IVec2 {
        let [x, y, z] = self.into();
        ivec2(x * 8 - z * 4, y * 8 - z * 4)
    }
}

impl TileSpace for IVec3 {}

/// Wrapper that imposes natural sorting ("a-10" after "a-9") for strings.
#[derive(Clone, Eq, PartialEq, Debug, Deserialize)]
struct Natural(String);

impl Ord for Natural {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        natord::compare(&self.0, &other.0)
    }
}

impl PartialOrd for Natural {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// vim:foldmethod=marker

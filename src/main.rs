#![allow(unused)]

use bevy::{
    prelude::*,
    render::{
        camera::{Camera, ScalingMode},
        render_resource::{
            Extent3d,
            TextureDescriptor,
            TextureDimension,
            TextureFormat,
            TextureUsages,
        },
        view::RenderLayers,
    },
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
};
use rand::{
    prelude::*,
    rngs::SmallRng,
};
use std::{
    collections::HashMap,
    convert::From,
};

const WINDOW_WIDTH: f32 = 1280.0;
const WINDOW_HEIGHT: f32 = 720.0;
const BLOCK_SIZE: u8 = 34; 
const COLUMNS: u8 = 10;
const ROWS: u8 = 20;

#[derive(Clone, Component, Copy, Debug)]
enum Tetronimo { I, O, T, L, J, S, Z }

impl Tetronimo {
    fn offsets(&self) -> &'static [(i8,i8); 4] {
        match self {
            Self::I => &[(0,0), (0,-1), (0,1), (0,2)], 
            Self::O => &[(0,0),  (1,0), (0,1), (1,1)],
            Self::T => &[(0,0), (-1,0), (1,0), (0,1)],
            Self::L => &[(0,0),  (0,2), (0,1), (1,0)],
            Self::J => &[(0,0), (-1,0), (0,2), (0,1)],
            Self::S => &[(0,0), (-1,1), (0,1), (1,0)],
            Self::Z => &[(0,0), (-1,0), (0,1), (1,1)]
        }
    }
}

impl From<u32> for Tetronimo {
    fn from(n: u32) -> Self {
        use Tetronimo::*;
        match n % 7 { 0 => I, 1 => O, 2 => T, 3 => L, 4 => J, 5 => S, 6 => Z, _ => unreachable!() }
    }
}

struct TetronimoStream {
    rng: SmallRng,
    next: Tetronimo,
}

impl TetronimoStream {
    fn new(mut rng: SmallRng) -> Self {
        let next = Tetronimo::from(rng.next_u32());
        Self { rng, next }
    }

    fn next(&mut self) -> Tetronimo {
        let next = self.next;
        self.next = Tetronimo::from(self.rng.next_u32());
        next
    }

    fn peek(&self) -> Tetronimo {
        self.next
    }
}

#[derive(Clone, Component, Copy, Debug)]
struct Child;

#[derive(Clone, Component, Copy, Debug)]
struct Preview;

#[derive(Clone, Component, Copy, Debug)]
struct PileRegion;

#[derive(Component, Debug)]
enum Orientation { R0, R90, R180, R270 }

impl Orientation {
    fn next(&self) -> Self {
        use Orientation::*;
        match self { R0 => R90, R90 => R180, R180 => R270, R270 => R0 }
    }
}

#[derive(Clone, Component, Copy, Debug, Default)]
struct Velocity(f32);

#[derive(Component, Debug)]
enum BlockType { Falling, Stacked }

#[derive(Clone, Component, Copy, Debug)]
struct Column(u8);

#[derive(Clone, Component, Copy, Debug)]
struct Row(u8);

#[derive(Debug, Event)]
struct CollisionEvent;

#[derive(Resource, Default)]
struct GameState {
    move_cooldown: Timer,
    velocity: Velocity,
    turbo: bool,
    tetronimo_stream: Option<TetronimoStream>,
}

fn setup_block_area_camera(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let (width, height) = ((COLUMNS as u32 * BLOCK_SIZE as u32), (ROWS as u32 * BLOCK_SIZE as u32));
    let size = Extent3d { width, height, ..default() };

    // create an image which will be the render surface for blocks
    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bgra8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING
                 | TextureUsages::COPY_DST
                 | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        ..default()
    };
    image.resize(size);  // clear the memory

    let image_handle = images.add(image);

    // create a camera that renders to this image
    commands.spawn((
        Camera2dBundle {
            camera: Camera { order: -1, target: image_handle.clone().into(), ..default() },
            projection: OrthographicProjection {
                near: -1000.0,
                far: 1000.0,
                ..default()
            },
            ..default()
        },
        RenderLayers::layer(1),
    ));

    // attach the image to a mesh so that it is displayed in the world
    commands.spawn((
        PileRegion,
        MaterialMesh2dBundle {
            mesh: Mesh2dHandle(meshes.add(Rectangle::new(width as f32, height as f32))),
            material: materials.add(ColorMaterial { texture: Some(image_handle), ..default() }),
            transform: Transform::from_xyz(
                WINDOW_WIDTH / 2.0,
                WINDOW_HEIGHT / 2.0,
                0.0
            ),
            ..default()
        },
        RenderLayers::layer(0),
    ));
}

fn setup_background(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn((
        MaterialMesh2dBundle {
            mesh: Mesh2dHandle(meshes.add(Rectangle::new(WINDOW_WIDTH, WINDOW_HEIGHT))),
            material: materials.add(Color::srgb(0.7, 0.7, 0.7)),
            transform: Transform::from_xyz(
                WINDOW_WIDTH / 2.0,
                WINDOW_HEIGHT / 2.0,
                -1.0
            ),
            ..default()
        },
        RenderLayers::layer(0),
    ));
    commands.spawn((
        MaterialMesh2dBundle {
            mesh: Mesh2dHandle(meshes.add(Rectangle::new(BLOCK_SIZE as f32 * 5.0, BLOCK_SIZE as f32 * 5.0))),
            material: materials.add(Color::srgb(0.0, 0.0, 0.0)),
            transform: Transform::from_xyz(
                WINDOW_WIDTH * 3.0 / 4.0,
                WINDOW_HEIGHT - BLOCK_SIZE as f32 * 3.5,
                -1.0
            ),
            ..default()
        },
        RenderLayers::layer(0),
    ));
}

fn setup_main_camera(
    mut commands: Commands,
) {
    commands.spawn((
        Camera2dBundle {
            projection: OrthographicProjection {
                scaling_mode: ScalingMode::FixedVertical(WINDOW_HEIGHT),
                near: -1000.0,
                far: 1000.0,
                ..default()
            },
            transform: Transform::from_xyz(WINDOW_WIDTH/2.0, WINDOW_HEIGHT/2.0, 0.0),
            ..default()
        },
        RenderLayers::layer(0),
    ));
}

fn fall(
    mut query: Query<(&mut Transform, &Velocity), With<Tetronimo>>,
    time: Res<Time>,
    game: Res<GameState>,
) {
    if let Ok((mut transform, Velocity(v))) = query.get_single_mut() {
        transform.translation.y -=
            time.delta().as_millis() as f32 * (if game.turbo { 1.0 } else { *v });
    }
}

fn spawn_tetronimo(
    query: Query<Entity, With<Tetronimo>>,
    preview_query: Query<Entity, With<Preview>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut game: ResMut<GameState>,
) {
    if let Ok(_) = query.get_single() {
        // do nothing
    } else {
        let tetronimo = game.tetronimo_stream.as_mut().unwrap().next();
        let velocity = game.velocity;
        
        // TODO 
        let square = Mesh2dHandle(meshes.add(Rectangle::new(BLOCK_SIZE as f32, BLOCK_SIZE as f32)));
        let color = materials.add(Color::srgb(0.0, 0.7, 0.7));

        commands.spawn((
            tetronimo,
            Orientation::R0,
            velocity,
            SpatialBundle::from_transform(Transform::from_xyz(
                BLOCK_SIZE as f32 / 2.0,
                ((ROWS+2) as f32 * BLOCK_SIZE as f32) / 2.0,
                0.0)
            ),
        )).with_children(move |parent| {
            for (dx,dy) in tetronimo.offsets() {
                parent.spawn((
                    Child,
                    BlockType::Falling,
                    RenderLayers::layer(1),
                    MaterialMesh2dBundle {
                        mesh: square.clone(),
                        material: color.clone(),
                        transform: Transform::from_xyz(BLOCK_SIZE as f32 * (*dx as f32),
                                                       BLOCK_SIZE as f32 * (*dy as f32),
                                                       0.0),
                        ..default()
                    },
                ));
            }
        });

        // change the preview
        if let Ok(preview_entity) = preview_query.get_single() {
            commands.entity(preview_entity).despawn_recursive();
        }
        let preview_tetronimo = game.tetronimo_stream.as_ref().unwrap().peek();
        let square = Mesh2dHandle(meshes.add(Rectangle::new(BLOCK_SIZE as f32, BLOCK_SIZE as f32)));
        let color = materials.add(Color::srgb(0.0, 0.7, 0.7));
        commands.spawn((
            Preview,
            SpatialBundle::from_transform(Transform::from_xyz(
                WINDOW_WIDTH * 3.0 / 4.0,
                WINDOW_HEIGHT - BLOCK_SIZE as f32 * 4.0,
                0.0)
            ),
        )).with_children(move |parent| {
            for (dx,dy) in preview_tetronimo.offsets() {
                parent.spawn((
                    RenderLayers::layer(0),
                    MaterialMesh2dBundle {
                        mesh: square.clone(),
                        material: color.clone(),
                        transform: Transform::from_xyz(BLOCK_SIZE as f32 * (*dx as f32),
                                                       BLOCK_SIZE as f32 * (*dy as f32),
                                                       0.0),
                        ..default()
                    },
                ));
            }
        });
    }
}

fn setup_pile(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let square = Mesh2dHandle(meshes.add(Rectangle::new(BLOCK_SIZE as f32, BLOCK_SIZE as f32)));
    let color = materials.add(Color::srgb(0.7, 0.0, 0.7));
    for row in 0..ROWS {
        for column in 0..COLUMNS {
            commands.spawn((
                Row(row),
                Column(column),
                MaterialMesh2dBundle {
                    mesh: square.clone(),
                    material: color.clone(),
                    transform: Transform::from_xyz(
                        BLOCK_SIZE as f32 * (column as f32 - COLUMNS as f32/2.0) + BLOCK_SIZE as f32 / 2.0,
                        BLOCK_SIZE as f32 * (row as f32 - ROWS as f32 / 2.0) + BLOCK_SIZE as f32 / 2.0, 
                        1.0
                    ),
                    visibility: Visibility::Hidden,
                    ..default()
                },
                RenderLayers::layer(1),
            ));
        }
    }
}

fn setup_game(
    mut game: ResMut<GameState>,
) {
    game.move_cooldown = Timer::from_seconds(0.05, TimerMode::Once);
    game.velocity = Velocity(0.05);
    game.tetronimo_stream = Some(TetronimoStream::new(SmallRng::seed_from_u64(8728347538)));
}

fn check_floor_collision(
    parent_query: Query<(Entity, &Transform), With<Tetronimo>>,
    children_query: Query<(Entity, &Transform), With<Child>>,
    pile_region: Query<&Transform, With<PileRegion>>,
    mut collision: EventWriter<CollisionEvent>,
) {
    if let Ok((parent, parent_transform)) = parent_query.get_single() {
        let parent_y = parent_transform.translation.y + pile_region.single().translation.y;
        for (child, child_transform) in &children_query {
            let child_y = parent_y + child_transform.translation.y;
            if child_y <= BLOCK_SIZE as f32 {
                collision.send(CollisionEvent);
                return;
            }
        }
    }
}

fn check_pile_collision(
    parent_query: Query<(Entity, &Transform), With<Tetronimo>>,
    children_query: Query<(Entity, &Transform), With<Child>>,
    pile_query: Query<(&Row, &Column, &Visibility)>,
    mut collision: EventWriter<CollisionEvent>,
) {
    if let Ok((parent_entity, parent_transform)) = parent_query.get_single() {
        let dx = parent_transform.translation.x + (BLOCK_SIZE as f32 * COLUMNS as f32) / 2.0;
        let dy = parent_transform.translation.y + (BLOCK_SIZE as f32 * ROWS as f32) / 2.0;

        for (_, child_transform) in &children_query {
            let (x,y) = (child_transform.translation.x + dx,
                         child_transform.translation.y + dy - BLOCK_SIZE as f32 / 2.0);

            let (col, row1, row2) = ((x / BLOCK_SIZE as f32).floor() as u8,
                                     (y / BLOCK_SIZE as f32).ceil() as u8,
                                     (y / BLOCK_SIZE as f32).floor() as u8);

            for (&Row(r), &Column(c), vis) in &pile_query {
                if vis == Visibility::Visible && (r == row1 || r == row2) && c == col {
                    collision.send(CollisionEvent);
                    return;
                }
            }
        }
    }
}

fn handle_collision(
    mut collision: EventReader<CollisionEvent>,
    parent_query: Query<(Entity, &Transform), With<Tetronimo>>,
    children_query: Query<(Entity, &Transform), With<Child>>,
    mut pile_query: Query<(&Row, &Column, &mut Visibility)>,
    mut commands: Commands,
) {
    if let Ok((parent_entity, parent_transform)) = parent_query.get_single() {
        let dx = parent_transform.translation.x + (BLOCK_SIZE as f32 * COLUMNS as f32) / 2.0;
        let dy = parent_transform.translation.y + (BLOCK_SIZE as f32 * ROWS as f32) / 2.0;
        for evt in collision.read() {
            // convert the falling blocks to pile blocks
            for (_, child_transform) in &children_query {

                let (x,y) = (child_transform.translation.x + dx,
                             child_transform.translation.y + dy);

                let (col, row) = ((x / BLOCK_SIZE as f32).floor() as u8,
                                  (y / BLOCK_SIZE as f32).floor() as u8);

                // check for game over
                if row >= ROWS {
                    // reset
                    for (_, _, mut vis) in &mut pile_query {
                        *vis = Visibility::Hidden;
                    }
                    commands.entity(parent_entity).despawn_recursive();
                    return;
                }

                for (&Row(r), &Column(c), mut vis) in &mut pile_query {
                    if r == row && c == col {
                        *vis = Visibility::Visible;
                    }
                }
            }

            // despawn the falling blocks
            let (entity, _) = parent_query.single();
            commands.entity(entity).despawn_recursive();

            // check for full rows in the pile
            let mut pile: [[bool; COLUMNS as usize]; ROWS as usize] = [[false; COLUMNS as usize]; ROWS as usize];

            for (&Row(r), &Column(c), vis) in &pile_query {
                pile[r as usize][c as usize] = vis == Visibility::Visible;
            }

            for row in 0..ROWS {
                let mut row_done = false;
                while !row_done {
                    let mut full = true; 
                    for col in 0..COLUMNS {
                        if !pile[row as usize][col as usize] {
                            full = false;
                            row_done = true;
                        }
                    }
                    if full {
                        for r in row..ROWS-1 {
                            pile[r as usize] = pile[(r+1) as usize];
                        }
                    }
                }
            }

            for (&Row(r), &Column(c), mut vis) in &mut pile_query {
                *vis = if pile[r as usize][c as usize] { Visibility::Visible } else { Visibility::Hidden };
            }
        }
    }
}

fn handle_input(
    mut parent_query: Query<(&mut Transform, &mut Orientation), With<Tetronimo>>,
    mut children_query: Query<(Entity, &mut Transform), (With<Child>, Without<Tetronimo>)>,
    pile_query: Query<(&Row, &Column, &Visibility)>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut game: ResMut<GameState>,
    time: Res<Time>,
) {
    let bound = (BLOCK_SIZE as f32 * COLUMNS as f32) / 2.0;
    if let Ok((mut parent_transform, mut orientation)) = parent_query.get_single_mut() {
        let dx = parent_transform.translation.x + (BLOCK_SIZE as f32 * COLUMNS as f32) / 2.0;
        let dy = parent_transform.translation.y + (BLOCK_SIZE as f32 * ROWS as f32) / 2.0;

        if game.move_cooldown.tick(time.delta()).finished() {
            let mut moved = false;

            if keyboard_input.just_pressed(KeyCode::ArrowLeft) {
                for (_, child_transform) in &children_query {
                    let (x,y) = (child_transform.translation.x + dx,
                                 child_transform.translation.y + dy);
                    let (col, row1, row2) = ((x / BLOCK_SIZE as f32).floor() as u8,
                                             (y / BLOCK_SIZE as f32).floor() as u8,
                                             (y / BLOCK_SIZE as f32).ceil() as u8);
                    if col == 0 {
                        return;
                    } else {
                        for (&Row(r), &Column(c), vis) in &pile_query {
                            if vis == Visibility::Visible && (r == row1 || r == row2) && col > 0 && c == (col-1) {
                                return;
                            }
                        }
                    }
                }

                parent_transform.translation.x =
                    f32::max(-bound, parent_transform.translation.x - BLOCK_SIZE as f32);
                moved = true;
            }

            if keyboard_input.just_pressed(KeyCode::ArrowRight) {
                for (_, child_transform) in &children_query {
                    let (x,y) = (child_transform.translation.x + dx,
                                 child_transform.translation.y + dy);
                    let (col, row1, row2) = ((x / BLOCK_SIZE as f32).floor() as u8,
                                             (y / BLOCK_SIZE as f32).floor() as u8,
                                             (y / BLOCK_SIZE as f32).ceil() as u8);
                    if col == COLUMNS-1 {
                        return;
                    } else {
                        for (&Row(r), &Column(c), vis) in &pile_query {
                            if vis == Visibility::Visible && (r == row1 || r == row2) && c == col+1 {
                                return;
                            }
                        }
                    }
                }

                parent_transform.translation.x =
                    f32::min(bound, parent_transform.translation.x + BLOCK_SIZE as f32);
                moved = true;
            }

            if keyboard_input.just_pressed(KeyCode::ArrowUp) {
                // first make sure rotation is possible
                let mut tmp = Vec::new();
                for (_, child_transform) in &children_query {
                    let (x,y) = (child_transform.translation.x,
                                 child_transform.translation.y);
                    let (x,y) = (-y + dx, x + dy);
                    let (col, row1, row2) = ((x / BLOCK_SIZE as f32).floor() as i16,
                                             (y / BLOCK_SIZE as f32).floor() as i16,
                                             (y / BLOCK_SIZE as f32).ceil() as i16);
                    tmp.push((col, row1));
                    tmp.push((col, row2));
                }
                for (col, row) in tmp {
                    if col < 0 || col >= COLUMNS as i16 {
                        return;
                    }
                    if row < 0 {
                        return;
                    }
                    for (&Row(r), &Column(c), vis) in &pile_query {
                        if vis == Visibility::Visible && r == row as u8 && c == col as u8 {
                            return;
                        }
                    }
                }

                // rotate
                *orientation = orientation.next();
                
                for (_, mut child_transform) in &mut children_query {
                    let (x, y) = (child_transform.translation.x, child_transform.translation.y);
                    child_transform.translation.x = -y;
                    child_transform.translation.y = x;
                }

                moved = true;
            }

            if keyboard_input.just_pressed(KeyCode::ArrowDown) {
                game.turbo = true;
            }

            if keyboard_input.just_released(KeyCode::ArrowDown) {
                game.turbo = false;
            }

            if moved {
                game.move_cooldown.reset();
            }
        }
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_event::<CollisionEvent>()
        .init_resource::<GameState>()
        .insert_resource(ClearColor(Color::srgb(0.0, 0.0, 0.0)))
        .add_systems(Startup, (
            setup_background,
            setup_block_area_camera,
            setup_main_camera,
            setup_pile,
            setup_game,
        ).chain())
        .add_systems(FixedUpdate, (
            fall,
            spawn_tetronimo,
            handle_input,
            check_floor_collision,
            check_pile_collision,
            handle_collision,
        ).chain())
        .run();
}


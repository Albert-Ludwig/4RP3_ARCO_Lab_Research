"""Generate a Mini fixture; contact geometry and dynamics are approximations."""
import xml.etree.ElementTree as ET


def add(parent, tag, text=None, **attrs):
    element = ET.SubElement(parent, tag, attrs)
    if text is not None:
        element.text = str(text)
    return element


def box(link, name, size, pose, mass):
    inertial = add(link, 'inertial')
    add(inertial, 'mass', mass)
    inertia = add(inertial, 'inertia')
    x, y, z = size
    for axis, value in [('ixx', mass*(y*y+z*z)/12),
                        ('iyy', mass*(x*x+z*z)/12), ('izz', mass*(x*x+y*y)/12)]:
        add(inertia, axis, value)
    add(inertial, 'pose', pose)
    collision = add(link, 'collision', name=name)
    add(collision, 'pose', pose)
    add(add(add(collision, 'geometry'), 'box'), 'size', ' '.join(map(str, size)))
    friction = add(add(add(collision, 'surface'), 'friction'), 'ode')
    add(friction, 'mu', 0.8)
    add(friction, 'mu2', 0.8)


def make_world(mesh, steering_efficiency=0.624):
    sdf = ET.Element('sdf', version='1.9')
    world = add(sdf, 'world', name='bunker_test')
    physics = add(world, 'physics', name='default', type='ignored')
    add(physics, 'max_step_size', 0.002)
    add(physics, 'real_time_factor', 1.0)
    for name, filename in [('Physics', 'physics'), ('UserCommands', 'user-commands'),
                           ('SceneBroadcaster', 'scene-broadcaster')]:
        plugin = add(world, 'plugin', name=f'gz::sim::systems::{name}',
                     filename=f'gz-sim-{filename}-system')
        if name == 'Physics':
            add(plugin, 'engine', filename='gz-physics-dartsim-plugin')
    light = add(world, 'light', name='sun', type='directional')
    add(light, 'pose', '0 0 10 0 0 0')
    add(light, 'diffuse', '0.8 0.8 0.8 1')
    add(light, 'direction', '-0.5 0.1 -0.9')
    ground = add(world, 'model', name='ground_plane')
    add(ground, 'static', 'true')
    ground_link = add(ground, 'link', name='ground')
    for tag in ('collision', 'visual'):
        plane = add(add(add(ground_link, tag, name=tag), 'geometry'), 'plane')
        add(plane, 'normal', '0 0 1')
        add(plane, 'size', '40 40')

    model = add(world, 'model', name='bunker_mini')
    add(model, 'pose', '0 0 0.28 0 0 0')
    base = add(model, 'link', name='base_link')
    # Upstream mesh bounds (m): x[-.333354,.359530], y[-.282,.282], z[-.274197,.056518].
    # Original whole-body mass: 40.953842 kg. Approximate split preserves this total.
    box(base, 'chassis', (0.5, 0.34, 0.15), '0 0 -0.07 0 0 0', 36.9538420525263)
    add(add(add(add(base, 'visual', name='manufacturer_mini_mesh'), 'geometry'), 'mesh'),
        'uri', mesh.as_uri())
    for name, y in [('left_track', 0.226), ('right_track', -0.226)]:
        track = add(model, 'link', name=name)
        add(track, 'pose', f'0 {y} -0.184197 0 0 0')
        box(track, 'track_contact', (0.64, 0.112, 0.18), '0 0 0 0 0 0', 2.0)
        joint = add(model, 'joint', name=name + '_fixed', type='fixed')
        add(joint, 'parent', 'base_link')
        add(joint, 'child', name)
        plugin = add(model, 'plugin', name='gz::sim::systems::TrackController',
                     filename='gz-sim-track-controller-system')
        add(plugin, 'link', name)
        add(plugin, 'max_command_age', 0.25)
        add(plugin, 'min_velocity', -0.6)
        add(plugin, 'max_velocity', 0.6)
    drive = add(model, 'plugin', name='gz::sim::systems::TrackedVehicle',
                filename='gz-sim-tracked-vehicle-system')
    add(add(drive, 'left_track'), 'link', 'left_track')
    add(add(drive, 'right_track'), 'link', 'right_track')
    # Harmonic 8.11's TrackedVehicle does not read the documented track_height tag.
    # Its internal 0.2 m virtual odometry wheel height cancels out in integration.
    for tag, value in [('tracks_separation', 0.452),
                       ('steering_efficiency', steering_efficiency), ('body_link', 'base_link'),
                       ('frame_id', 'odom'), ('child_frame_id', 'base_link'),
                       ('odom_publish_frequency', 50)]:
        add(drive, tag, value)
    for tag, limit in [('linear_velocity', 0.35), ('angular_velocity', 0.6)]:
        limiter = add(drive, tag)
        add(limiter, 'min_velocity', -limit)
        add(limiter, 'max_velocity', limit)
    truth = add(model, 'plugin', name='gz::sim::systems::OdometryPublisher',
                filename='gz-sim-odometry-publisher-system')
    for tag, value in [('odom_frame', 'world'), ('robot_base_frame', 'base_link'),
                       ('odom_topic', '/model/bunker_mini/ground_truth/odometry'),
                       ('tf_topic', '/model/bunker_mini/ground_truth/tf'),
                       ('odom_publish_frequency', 50), ('dimensions', 3)]:
        add(truth, tag, value)
    ET.indent(sdf)
    return ET.tostring(sdf, encoding='unicode')


def make_description(mesh):
    robot = ET.Element('robot', name='bunker_mini')
    base = add(robot, 'link', name='base_link')
    add(add(add(base, 'visual'), 'geometry'), 'mesh', filename=mesh.as_uri())
    for name, y in [('left_track', 0.226), ('right_track', -0.226)]:
        add(robot, 'link', name=name)
        joint = add(robot, 'joint', name=name + '_fixed', type='fixed')
        add(joint, 'parent', link='base_link')
        add(joint, 'child', link=name)
        add(joint, 'origin', xyz=f'0 {y} -0.184197', rpy='0 0 0')
    return ET.tostring(robot, encoding='unicode')

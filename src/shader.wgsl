// Vertex shader

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,

    @location(9) normal_matrix_0: vec3<f32>,
    @location(10) normal_matrix_1: vec3<f32>,
    @location(11) normal_matrix_2: vec3<f32>,
};

struct Camera {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}
@group(1) @binding(0)
var<uniform> camera: Camera;

@group(1) @binding(1)
var<uniform> light: Camera;


struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_position: vec3<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32> (
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    let normal_matrix = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2,
    );

    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.world_normal = normal_matrix * model.normal;

    var world_position: vec4<f32> = model_matrix * vec4<f32>(model.position, 1.0);
    out.world_position = world_position.xyz;
    out.clip_position = camera.view_proj * world_position;
    return out;
}


// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@group(1) @binding(2)
var t_shadow: texture_depth_2d;
@group(1) @binding(3)
var s_shadow: sampler_comparison;

// TODO CITE: wgpu/examples/shadow
fn fetch_shadow(homogeneous_coords: vec4<f32>) -> f32 {
    if (homogeneous_coords.w <= 0.0) {
        return 1.0;
    }
    // compensate for the Y-flip difference between the NDC and texture coordinates
    let flip_correction = vec2<f32>(0.5, -0.5);
    // compute texture coordinates for shadow lookup
    let proj_correction = 1.0 / homogeneous_coords.w;
    let light_local = homogeneous_coords.xy * flip_correction * proj_correction + vec2<f32>(0.5, 0.5);
    // do the lookup, using HW PCF and comparison
    return textureSampleCompareLevel(t_shadow, s_shadow, light_local, homogeneous_coords.z * proj_correction);
}

@fragment
fn fs_dummy(in: VertexOutput, @builtin(front_facing) face: bool) -> @location(0) vec4<f32> {
    let object_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coords);

    return object_color;
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) face: bool) -> @location(0) vec4<f32> {
    let light_color = vec3<f32>(1.0, 1.0, 1.0);

    let object_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coords);

    let ambient_strength = 0.1;
    let ambient_color = light_color * ambient_strength;

    let light_dir = normalize(light.view_pos.xyz - in.world_position.xyz);
    let diffuse_strength = max(dot(in.world_normal, light_dir), 0.0);
    let diffuse_color = light_color * diffuse_strength;

    let view_dir = normalize(camera.view_pos.xyz - in.world_position.xyz);
    let half_dir = normalize(view_dir + light_dir);

    // https://github.com/mcclure/webgpu-tutorial-rs/blob/webgpu-tutorial/src/shader.wgsl
    // This one-dimensional separable blur filter samples five points and averages them by different amounts.
    // If we do it on two separate axes, we get a 2d blur.
    // Weights and offsets taken from http://rastergrid.com/blog/2010/09/efficient-gaussian-blur-with-linear-sampling/

    // The weights for the center, one-point-out, and two-point-out samples
    const WEIGHT0 = 0.2270270270;
    const WEIGHT1 = 0.3162162162;
    const WEIGHT2 = 0.0702702703;

    // The distances-from-center for the samples
    const OFFSET1 = 1.3846153846;
    const OFFSET2 = 3.2307692308;

    let blur_resolution = vec2<f32>(60.0, 60.0);

    var shadow_guassian = 0.0;
    shadow_guassian += fetch_shadow(light.view_proj* vec4<f32>(in.world_position, 1.0)) * WEIGHT0;
    shadow_guassian += fetch_shadow(light.view_proj* vec4<f32>(in.world_position.xy + blur_resolution * OFFSET1, in.world_position.z, 1.0)) * WEIGHT1;
    shadow_guassian += fetch_shadow(light.view_proj* vec4<f32>(in.world_position.xy - blur_resolution * OFFSET1, in.world_position.z, 1.0)) * WEIGHT1;
    shadow_guassian += fetch_shadow(light.view_proj* vec4<f32>(in.world_position.xy + blur_resolution * OFFSET2, in.world_position.z, 1.0)) * WEIGHT2;
    shadow_guassian += fetch_shadow(light.view_proj* vec4<f32>(in.world_position.xy - blur_resolution * OFFSET2, in.world_position.z, 1.0)) * WEIGHT2;

    let shadow_strength = 0.7;
    let shadow = fetch_shadow(light.view_proj * vec4<f32>(in.world_position, 1.0));


    let specular_strength = pow(max(dot(in.world_normal, half_dir), 0.0), 32.0);

    // Disable specular effects in shadow
    let specular_color = specular_strength * light_color * shadow;

    let result = (ambient_color + diffuse_color + specular_color) * (shadow_guassian * shadow_strength + (1.0 - shadow_strength)) * object_color.xyz;
    // let result = (ambient_color + diffuse_color + specular_color) * (shadow_guassian * shadow_strength + (1.0 - shadow_strength));

    return vec4<f32>(result, object_color.a);
}
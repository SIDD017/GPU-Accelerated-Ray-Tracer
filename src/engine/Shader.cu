#include "Shader.cuh"

namespace App {

/** Checks if shader compilation succeeded. */
void check_shader_compilation(const unsigned int ID, std::string shader_type) {
    int success;
    char info_log[512];
    glGetShaderiv(ID, GL_COMPILE_STATUS, &success);
    if(!success) {
        glGetShaderInfoLog(ID, 512, NULL, info_log);
        std::cout << "ERROR::SHADER" + shader_type + "::COMPILATION_FAILED\n" << info_log << std::endl;
    }
}

/** Checks if shader compilation succeeded. */
void check_shader_linking(const unsigned int ID) {
    int success;
    char info_log[512];
    glGetProgramiv(ID, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(ID, 512, NULL, info_log);
		std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << info_log << std::endl;
	}
}

Shader::Shader(const char *vertex_path, const char *fragment_path, const char *geometry_path) {

    std::string vertex_code;
    std::string geometry_code;
    std::string fragment_code;
    std::ifstream vertex_file;
    std::ifstream geometry_file;
    std::ifstream fragment_file;

    // ensure ifstream objects can throw exceptions:
	vertex_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	geometry_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	fragment_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    // Read GLSL code from shader files
    try {
        std::cout << vertex_path << std::endl;
        std::cout << fragment_path << std::endl;
        vertex_file.open(vertex_path);
        fragment_file.open(fragment_path);
        std::stringstream v_shader_stream, g_shader_stream, f_shader_stream;
        v_shader_stream << vertex_file.rdbuf();
        f_shader_stream << fragment_file.rdbuf();
        vertex_file.close();
        fragment_file.close();

        vertex_code = v_shader_stream.str();
        fragment_code = f_shader_stream.str();

        if(geometry_path != nullptr) {
            geometry_file.open(geometry_path);
            g_shader_stream << geometry_file.rdbuf();
            geometry_file.close();
            geometry_code = g_shader_stream.str();
        }
    }
    catch (std::ifstream::failure e) {
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ\n";
    }

    const char *v_shader_code = vertex_code.c_str();
    const char *f_shader_code = fragment_code.c_str();
    // IDs for vertex, geometry and fragment shader objects
    unsigned int vertex, geometry, fragment;

    /* Vertex Shader */
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &v_shader_code, NULL);
    glCompileShader(vertex);
    check_shader_compilation(vertex,"VERTEX");

    /* Fragment Shader */
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &f_shader_code, NULL);
    glCompileShader(fragment);
    check_shader_compilation(fragment,"FRAGMENT");

    /* Geometry shader, if it exists. */
    if(geometry_path != nullptr) {
        const char *g_shader_code = geometry_code.c_str();
        geometry = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(geometry, 1, &g_shader_code, NULL);
        glCompileShader(geometry);
        check_shader_compilation(geometry, "GEOMETRY");
    }

    /* Link shaders and create the final shader program. */
    ID = glCreateProgram();
    glAttachShader(ID, vertex);
    if(geometry_path != nullptr) {
        glAttachShader(ID, geometry);
    }
    glAttachShader(ID, fragment);
    glLinkProgram(ID);
    check_shader_linking(ID);

    /* Once shader program is created, component shaders can be deleted. */
    glDeleteShader(vertex);
    if(geometry_path != nullptr) {
        glDeleteShader(geometry);
    }
    glDeleteShader(fragment);
    
}

void Shader::use_shader() {
    glUseProgram(ID);
}

void Shader::delete_shade_program() {
    glDeleteProgram(ID);
}

void Shader::set_bool(const std::string &name, bool value) const {
    glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}

void Shader::set_int(const std::string &name, int value) const {
    glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}

void Shader::set_float(const std::string &name, float value) const {
    glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
}

void Shader::set_vec2(const std::string &name, float value[2]) const {
    glUniform2f(glGetUniformLocation(ID, name.c_str()), value[0], value[1]);
}

void Shader::set_vec3(const std::string &name, float value[3]) const {
    glUniform3f(glGetUniformLocation(ID, name.c_str()), value[0], value[1], value[2]);
}

void Shader::set_vec4(const std::string &name, float value[4]) const {
    glUniform4f(glGetUniformLocation(ID, name.c_str()), value[0], value[1], value[2], value[3]);
}

}
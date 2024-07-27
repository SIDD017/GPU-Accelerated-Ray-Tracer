#ifndef APP_SHADER_H
#define APP_SHADER_H

#include <glad/glad.h>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

namespace App {

class Shader {
private:
/* ID for Shader program object. */
    unsigned int ID;
public:
    Shader(const char *vertex_path, const char *fragment_path, const char *geometry_path = nullptr);
    /* Use the shader program linked to this object. */
    void use_shader();
    /* Utility functions to set uniforms */
    void set_bool(const std::string &name, bool value) const;
    void set_int(const std::string &name, int value) const;
    void set_float(const std::string &name, float value) const;
    void set_vec2(const std::string &name, float value[2]) const;
    void set_vec3(const std::string &name, float value[3]) const;
    void set_vec4(const std::string &name, float value[4]) const;
    /* Deletes the shader program*/
    void delete_shade_program();
};

}

#endif
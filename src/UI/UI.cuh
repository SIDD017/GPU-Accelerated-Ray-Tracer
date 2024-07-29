#ifndef APP_UI_H
#define APP_UI_H

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "imgui.h"
#include "imgui_internal.h"

namespace App {
class UI {
private:
public:
  UI(GLFWwindow *window);
  ~UI();

  void overlay();
};
} // namespace App
#endif
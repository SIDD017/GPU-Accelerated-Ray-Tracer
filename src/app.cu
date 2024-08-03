#include "app.cuh"
#include <cuda_gl_interop.h>

namespace App {

Engine::Engine() {

  context = new Context;

  /* Window Manager */
  if (!glfwInit()) {
    std::cout << "ERROR: Unable to initialize GLFW";
    return;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  context->window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "GPUTracer", NULL, NULL);

  if (!(context->window)) {
    std::cout << "ERROR: Failed to create GLFW window";
    glfwTerminate();
    return;
  }

  /* Make the window's context as the main context on the current thread. */
  glfwMakeContextCurrent(context->window);

  /* Set IO callbacks. */
  glfwSetFramebufferSizeCallback(context->window, framebuffer_size_callback);

  /* Initialize GLAD */
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "ERROR: Failed to initialize GLAD";
    return;
  }

  context->UI = new UI(context->window);
}

Engine::~Engine() { glfwTerminate(); }

/* Resizing window */
void Engine::framebuffer_size_callback(GLFWwindow *window, int width,
                                       int height) {
  glViewport(0, 0, width, height);
}

/* Callback to process input events */
void Engine::processInput(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, true);
  }
}

void Engine::init_shaders() {
  shaders = new Shader("shaders/vert.glsl", "shaders/frag.glsl");

  float vertices[] = {
    -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,  
    -1.0f,  1.0f, 0.0f, 0.0f, 1.0f, 
     1.0f, -1.0f, 0.0f, 1.0f, 0.0f,

    -1.0f,  1.0f, 0.0f, 0.0f, 1.0f, 
     1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
     1.0f, -1.0f, 0.0f, 1.0f, 0.0f
  };

  unsigned int VBO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

void Engine::draw() {
  tracer->draw(8, 8, cgr);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SCR_WIDTH, SCR_HEIGHT, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
  glGenerateMipmap(GL_TEXTURE_2D);
  
  shaders->use_shader();
  glBindVertexArray(VAO);
  glDrawArrays(GL_TRIANGLES, 0, 6);
}

void Engine::execute() {

  init_shaders();
  tracer = new CUDA_Tracer::Tracer(SCR_WIDTH, SCR_HEIGHT, 10);
  
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  // load and generate the texture
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glBindTexture(GL_TEXTURE_2D, 0);

  unsigned int PBO;
  glGenBuffers(1, &PBO);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, SCR_WIDTH * SCR_HEIGHT * 4, NULL, GL_DYNAMIC_COPY);
  cudaGraphicsGLRegisterBuffer(&cgr, PBO, cudaGraphicsRegisterFlagsNone);

  /* Main Render loop */
  while (!glfwWindowShouldClose(context->window)) {

    context->UI->overlay();

    /* Render here */
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(0.0f, 1.0f, 0.0f, 1.0f);

    /* If draw callback is not NULL, the render the scene */
    draw();

    /* If scene graph is not NULL then render the scene */

    /* ImGui Render */
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    /* Swap front and back buffers */
    glfwSwapBuffers(context->window);

    /* Poll for and process events */
    glfwPollEvents();
  }

  /* Cleanup all allocated memory to prevent leaks */
  delete shaders;
}

} // namespace App
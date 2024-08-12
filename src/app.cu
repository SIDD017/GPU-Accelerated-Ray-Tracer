#include "app.cuh"
#include <cuda_gl_interop.h>

namespace App {

float Engine::lastX = SCR_WIDTH / 2.0f;
float Engine::lastY = SCR_HEIGHT / 2.0f;
bool Engine::first_mouse = true;
CUDA_Tracer::camera_properties Engine::cam;

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
  glfwSetCursorPosCallback(context->window, mouse_callback);

  /* Initialize GLAD */
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "ERROR: Failed to initialize GLAD";
    return;
  }

  context->UI = new UI(context->window);
  deltaTime = 0.0f;
  lastFrame = 0.0f;
}

Engine::~Engine() { glfwTerminate(); }

/* Resizing window */
void Engine::framebuffer_size_callback(GLFWwindow *window, int width,
                                       int height) {
  glViewport(0, 0, width, height);
}

void Engine::ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true)
{
    xoffset *= cam.sensitivity;
    yoffset *= cam.sensitivity;

    cam.Yaw   += xoffset;
    cam.Pitch += yoffset;

    // make sure that when pitch is out of bounds, screen doesn't get flipped
    if (constrainPitch)
    {
        if (cam.Pitch > 89.0f)
            cam.Pitch = 89.0f;
        if (cam.Pitch < -89.0f)
            cam.Pitch = -89.0f;
    }

    // update Front, Right and Up Vectors using the updated Euler angles
    updateCameraVectors();
}

void Engine::updateCameraVectors()
{
    // calculate the new Front vector
    float x = cos(glm::radians(cam.Yaw)) * cos(glm::radians(cam.Pitch));
    float y = sin(glm::radians(cam.Pitch));
    float z = sin(glm::radians(cam.Yaw)) * cos(glm::radians(cam.Pitch));
    CUDA_Tracer::vec3 front(x, y, z);
    cam.look_at = cam.look_from + front;
    // also re-calculate the Right and Up vector
    CUDA_Tracer::vec3 right = CUDA_Tracer::cross(front, CUDA_Tracer::vec3(0,1,0));  // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
    cam.vup = CUDA_Tracer::unit_vector(CUDA_Tracer::cross(right, front));
}

void Engine::mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (first_mouse)
    {
        lastX = xpos;
        lastY = ypos;
        first_mouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    ProcessMouseMovement(xoffset, yoffset);
}

// void updateCameraVectors()
// {
//   float velocity = MovementSpeed * deltaTime;
//   if (direction == FORWARD)
//       Position += Front * velocity;
//   if (direction == BACKWARD)
//       Position -= Front * velocity;
//   if (direction == LEFT)
//       Position -= Right * velocity;
//   if (direction == RIGHT)
//       Position += Right * velocity;
// }

/* Callback to process input events */
void Engine::processInput(GLFWwindow *window, CUDA_Tracer::camera_properties *cam, float deltaTime) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, true);
  }
  float velocity = cam->movement_speed * deltaTime;
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
    CUDA_Tracer::vec3 Front = CUDA_Tracer::unit_vector(cam->look_at - cam->look_from);
    cam->look_from += Front * velocity;
    cam->look_at += Front * velocity;
  }
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    CUDA_Tracer::vec3 Front = CUDA_Tracer::unit_vector(cam->look_at - cam->look_from);
    cam->look_from -= Front * velocity;
    cam->look_at -= Front * velocity;
  }
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
    CUDA_Tracer::vec3 Right = CUDA_Tracer::unit_vector(CUDA_Tracer::cross((cam->look_at - cam->look_from), cam->vup));
    cam->look_from -= Right * velocity;
    cam->look_at -= Right * velocity;
  }
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
    CUDA_Tracer::vec3 Right = CUDA_Tracer::unit_vector(CUDA_Tracer::cross((cam->look_at - cam->look_from), cam->vup));
    cam->look_from += Right * velocity;
    cam->look_at += Right * velocity;
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

void Engine::draw(CUDA_Tracer::camera_properties &cam) {
  processInput(context->window, &cam, deltaTime);
  tracer->draw(32, 32, cgr, cam);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SCR_WIDTH, SCR_HEIGHT, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
  glGenerateMipmap(GL_TEXTURE_2D);
  
  shaders->use_shader();
  glBindVertexArray(VAO);
  glDrawArrays(GL_TRIANGLES, 0, 6);
}

void Engine::execute() {

  init_shaders();
  tracer = new CUDA_Tracer::Tracer(SCR_WIDTH, SCR_HEIGHT, 8);
  
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

  cam.look_at = CUDA_Tracer::vec3(0,0,-1);
  cam.look_from = CUDA_Tracer::vec3(3,3,2);
  cam.aperture = 0.1;
  cam.vup = CUDA_Tracer::vec3(0,1,0);
  cam.vfov = 20.0;
  cam.movement_speed = 2.5f;
  cam.sensitivity = 0.4f;
  cam.dist_to_focus = (cam.look_from-cam.look_at).length();

  /* Main Render loop */
  while (!glfwWindowShouldClose(context->window)) {
    float currentFrame = static_cast<float>(glfwGetTime());
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;
    context->UI->overlay();

    /* Render here */
    glClear(GL_COLOR_BUFFER_BIT);
    glClearColor(0.0f, 1.0f, 0.0f, 1.0f);

    /* If draw callback is not NULL, the render the scene */
    draw(cam);

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
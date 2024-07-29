#include "app.cuh"

int main() {

  App::Engine *engine = new App::Engine();
  engine->execute();

  return 0;
}
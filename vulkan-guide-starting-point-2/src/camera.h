
#include <vk_types.h>
#include <SDL_events.h>

class Camera {
public:

	bool isActive = true;

	glm::vec3 velocity;
	glm::vec3 position;

	float pitch{ 0.f }; //vertical
	float yaw{ 0.f };	//horizontal

	glm::mat4 getViewMatrix();
	glm::mat4 getRotationMatrix();

	void processSDLEvent(SDL_Event& e);

	void update();
};


#include <vk_types.h>
#include <SDL_events.h>

enum CameraType
{
	Follow = 0,
	Orbit
};

class Camera {
public:

	bool isActive = true;

	glm::vec3 velocity;
	glm::vec3 position;

	float pitch{ 0.f }; //vertical
	float yaw{ 0.f };	//horizontal

	float distanceToTarget = 200.0f;
	glm::vec3 target = glm::vec3(0.0f);

	glm::mat4 getViewMatrix();
	glm::mat4 getRotationMatrix();

	void processSDLEvent(SDL_Event& e);

	void update();

	CameraType cameraType = Orbit;
};

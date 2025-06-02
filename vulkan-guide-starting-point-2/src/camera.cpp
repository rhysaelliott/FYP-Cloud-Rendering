#include "camera.h"
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

glm::mat4 Camera::getViewMatrix()
{
	if (cameraType == Orbit)
	{
		glm::mat4 rot = getRotationMatrix();
		glm::vec3 offset = glm::vec3(rot * glm::vec4(0.0f, 0.0f, distanceToTarget, 0.0f));
		position = target + offset;

		return glm::lookAt(position, target, glm::vec3(0, 1, 0));
	}

	glm::mat4 cameraTranslation = glm::translate(glm::mat4(1.f), position);
	glm::mat4 cameraRotation = getRotationMatrix();
	return glm::inverse(cameraTranslation * cameraRotation);
}
glm::mat4 Camera::getRotationMatrix()
{
	glm::quat pitchRotation = glm::angleAxis(pitch, glm::vec3{ 1.f,0.f,0.f });
	glm::quat yawRotation = glm::angleAxis(yaw, glm::vec3{ 0.f,-1.f,0.f });

	return glm::toMat4(yawRotation) * glm::toMat4(pitchRotation);
}

void Camera::processSDLEvent(SDL_Event& e)
{
	if (!isActive) return;
	if (cameraType == CameraType::Orbit)
	{
		if (e.type == SDL_MOUSEMOTION)
		{
			yaw += (float)e.motion.xrel / 2000.f;
			pitch -= (float)e.motion.yrel / 2000.f;

			pitch = glm::clamp(pitch, -glm::half_pi<float>() + 0.01f, glm::half_pi<float>() - 0.01f);
		}
		if (e.type == SDL_MOUSEWHEEL)
		{
			distanceToTarget -= e.wheel.y;
			distanceToTarget = glm::clamp(distanceToTarget, 1.0f, 1000.0f);
		}
		return;
	}
	if (e.type == SDL_KEYDOWN)
	{
		if (e.key.keysym.sym == SDLK_w) { velocity.z = -1; }
		if (e.key.keysym.sym == SDLK_s) { velocity.z = 1; }
		if (e.key.keysym.sym == SDLK_a) { velocity.x = -1; }
		if (e.key.keysym.sym == SDLK_d) { velocity.x = 1; }
		if (e.key.keysym.sym == SDLK_q) { velocity.y = -1; }
		if (e.key.keysym.sym == SDLK_e) { velocity.y = 1; }
	}
	if (e.type == SDL_KEYUP)
	{
		if (e.key.keysym.sym == SDLK_w) { velocity.z = 0; }
		if (e.key.keysym.sym == SDLK_s) { velocity.z = 0; }
		if (e.key.keysym.sym == SDLK_a) { velocity.x = 0; }
		if (e.key.keysym.sym == SDLK_d) { velocity.x = 0; }
		if (e.key.keysym.sym == SDLK_q) { velocity.y  = 0; }
		if (e.key.keysym.sym == SDLK_e) { velocity.y = 0; }
	}

	if (e.type == SDL_MOUSEMOTION)
	{
		yaw += (float)e.motion.xrel / 200.f;
		pitch -= (float)e.motion.yrel / 200.f;
	}
}

void Camera::update()
{
	if (!isActive || cameraType == CameraType::Orbit) return;
	glm::mat4 cameraRotation = getRotationMatrix();
	position += glm::vec3(cameraRotation * glm::vec4(velocity, 0.f));

}
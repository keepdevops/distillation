/**
 * Simple vanilla knowledge distillation in C++ (LibTorch).
 * Bare-metal, air-gapped. M3 MPS support.
 *
 * Prerequisites:
 * - Download LibTorch ARM64 from pytorch.org
 * - Export teacher/student from Python via torch.jit.trace
 *
 * Build: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch -B build && cmake --build build
 * Run:   ./build/distill
 */

#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>

const float temperature = 5.0f;
const float alpha = 0.5f;
const int epochs = 3;
const int batch_size = 8;

int main() {
  torch::Device device = torch::kCPU;
#if defined(TORCH_BACKEND_MPS)
  if (torch::mps::is_available()) {
    device = torch::kMPS;
    std::cout << "Using MPS (Apple Silicon)" << std::endl;
  }
#endif
  if (device.is_cpu())
    std::cout << "Using CPU" << std::endl;

  // Load TorchScript models (export from Python first)
  const char* teacher_path = std::getenv("DISTILL_TEACHER_PATH");
  const char* student_path = std::getenv("DISTILL_STUDENT_PATH");
  if (!teacher_path) teacher_path = "teacher.pt";
  if (!student_path) student_path = "student.pt";

  std::cout << "[distill] Loading teacher: " << teacher_path << std::endl;
  std::cout << "[distill] Loading student: " << student_path << std::endl;

  torch::jit::Module teacher;
  torch::jit::Module student;
  try {
    teacher = torch::jit::load(teacher_path);
    student = torch::jit::load(student_path);
  } catch (const c10::Error& e) {
    std::cerr << "[distill][ERROR] Model load failed: " << e.what() << std::endl;
    std::cerr << "[distill] Export models from Python:\n"
              << "  torch.jit.trace(teacher, example_input).save('teacher.pt')\n"
              << "  torch.jit.trace(student, example_input).save('student.pt')\n"
              << "  Set DISTILL_TEACHER_PATH and DISTILL_STUDENT_PATH if needed.\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "[distill][ERROR] Unexpected error: " << e.what() << std::endl;
    return 1;
  }

  teacher.to(device);
  student.to(device);
  teacher.eval();
  student.train();

  torch::optim::Adam optimizer(student.parameters(), 0.001);

  for (int epoch = 0; epoch < epochs; epoch++) {
    float total_loss = 0.0f;
    // Toy loop: use random data (replace with HDF5/CSV loader for real data)
    for (int b = 0; b < 10; b++) {
      auto inputs = torch::randn({batch_size, 3, 32, 32}).to(device);
      auto labels = torch::randint(0, 10, {batch_size}).to(device);

      torch::NoGradGuard no_grad;
      auto teacher_out = teacher.forward({inputs}).toTensor();
      auto student_out = student.forward({inputs}).toTensor();

      auto soft_teacher = torch::softmax(teacher_out / temperature, 1);
      auto soft_student = torch::log_softmax(student_out / temperature, 1);
      // Use Sum then /batch_size for batchmean (Reduction::BatchMean was removed in newer LibTorch)
      auto kl = torch::kl_div(soft_student, soft_teacher,
                              torch::Reduction::Sum) / batch_size * (temperature * temperature);
      auto ce = torch::nll_loss(torch::log_softmax(student_out, 1), labels);
      auto loss = alpha * ce + (1.0f - alpha) * kl;

      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
      total_loss += loss.item<float>();
    }
    std::cout << "Epoch " << epoch << " Loss: " << total_loss << std::endl;
  }

  student.save("distilled_student.pt");
  std::cout << "Saved distilled_student.pt" << std::endl;
  return 0;
}

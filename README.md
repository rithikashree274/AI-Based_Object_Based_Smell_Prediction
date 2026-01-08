This project proposes an AI-based system that predicts probable smells and associated chemical compounds using visual object detection and contextual inference, without relying on physical smell sensors.
The system takes detected objects from an environment (such as vehicles, roads, people, or traffic signals) along with an intensity value and uses a machine learning model to infer the most likely smell category. Based on the predicted smell, the system then maps it to a corresponding chemical compound and chemical type using a predefined chemical knowledge base.
Instead of directly detecting chemical molecules which requires specialized hardware sensors the project focuses on software based smell estimation using historical and contextual data patterns. The model is trained on a labeled dataset that represents common real-world associations between objects, smells, and chemicals (for example, vehicles with exhaust gases, roads with dust particles, or humans with body odors).
A Random Forest classification model is used to learn these patterns due to its robustness, ability to handle non-linear relationships, and good generalization on tabular data. The trained model outputs:
1. Detected object
2. Intensity level
3. Predicted smell
4. Associated chemical name
5. Chemical type
The final system also reports the overall model accuracy and displays sample predictions to demonstrate performance.
This approach is particularly useful in scenarios where hardware smell sensors are unavailable, costly, or impractical, such as large-scale surveillance, smart cities, traffic monitoring, or preliminary environmental assessment systems.

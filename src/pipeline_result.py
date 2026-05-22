from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self):
        return self.x + self.width

    @property
    def y2(self):
        return self.y + self.height


@dataclass(frozen=True)
class Annotation:
    label: str
    bbox: BoundingBox
    confidence: float | None = None
    category: str = "object"

    def to_dict(self):
        payload = asdict(self)
        payload["bbox"]["x2"] = self.bbox.x2
        payload["bbox"]["y2"] = self.bbox.y2
        return payload


@dataclass
class PipelineResult:
    image: object
    annotations: list[Annotation] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)

    def to_export_dict(self, pipeline, params, image_shape):
        height, width = image_shape[:2]
        return {
            "pipeline": {
                "id": pipeline.id,
                "name": pipeline.name,
                "category": pipeline.category,
                "task_type": pipeline.task_type,
                "status": pipeline.status,
            },
            "input": {
                "width": int(width),
                "height": int(height),
            },
            "params": params,
            "labels": self.labels,
            "metrics": self.metrics,
            "annotations": [annotation.to_dict() for annotation in self.annotations],
            "messages": self.messages,
        }

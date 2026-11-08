from uuid import uuid4

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class LeafPrediction(Base):
    __tablename__ = "leaf_prediction"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid4()))
    leaf_id: Mapped[str] = mapped_column(ForeignKey("leaf.id"), nullable=False, index=True)
    picture_url: Mapped[str] = mapped_column(String, nullable=False)

    leaf: Mapped["Leaf"] = relationship(back_populates="predictions")

    def get_feature(self, feature_name: str):
        """Busca uma característica na folha associada à previsão."""
        if self.leaf is None:
            return None
        return self.leaf.get_feature(feature_name)

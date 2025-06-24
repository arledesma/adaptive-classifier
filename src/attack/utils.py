from datetime import datetime
from dataclasses import dataclass, field
import logging
from pathlib import Path
from pyattck import Attck
from pyattck_data.tactic import Tactic
from pyattck_data.technique import Technique


@dataclass
class AttackUtility:
    data_path: Path
    _last_update: datetime | None = field(default=None, init=False)
    __attack_data: Attck | None = field(default=None, init=False)
    debug: bool = field(default=False, kw_only=True)

    def __post_init__(self):
        logging.basicConfig(level=logging.INFO if not self.debug else logging.DEBUG)
        self.logger = logging.getLogger("AttackUtility")

    def setup_attack_cache_path(self, data_path: Path | None = None) -> "AttackUtility":
        """
        Set up the attack cache path for storing attack data.
        """
        if data_path is None:
            # Use the default attack cache path if no path is provided
            data_path = AttackUtility.get_attck_cache_path()
        data_path.mkdir(parents=True, exist_ok=True)
        self.data_path = data_path
        self._last_update = datetime.now()
        return self

    def update_attck(self) -> "AttackUtility":
        if self.__attack_data is None:
            self.__get_attack_data()
        else:
            self.__attack_data.update()
        self._last_update = datetime.now()
        return self

    @staticmethod
    def get_attck_cache_path() -> Path:
        """
        Get the path to the attack cache directory.
        This directory is used to store attack data.
        """
        home_dir = Path.home()
        data_path = home_dir / ".cache" / "attack_data"
        return data_path

    def __get_techniques(self, techniques: list[Technique]) -> list[dict]:
        simple_techniques: list[dict] = []
        for technique in techniques:
            if not isinstance(technique, Technique):
                continue
            technique_id: str = "unknown" if not technique.technique_id else technique.technique_id
            technique_name: str = "unknown" if not technique.name else technique.name
            technique_description: str = "unknown" if not technique.description else technique.description
            simple_techniques.append({
                "technique_id": technique_id,
                "name": technique_name,
                "description": technique_description
            })
        return simple_techniques

    def __get_tactics(self, tactics: list[Tactic]) -> list[dict]:
        simple_tactics_with_techniques = []
        for tactic in tactics:
            if not isinstance(tactic, Tactic):
                continue

            tactic_id: str
            if tactic.external_references and len(tactic.external_references) > 0:
                tactic_id = tactic.external_references[0].external_id
            else:
                tactic_id = "unknown"

            techniques: list = self.__get_techniques(tactic.techniques)

            tac = {
                'tactic_id': tactic_id,
                'name': tactic.name,
                'description': tactic.description,
                'techniques': techniques
            }
            simple_tactics_with_techniques.append(tac)
        return simple_tactics_with_techniques

    def __get_attack_data(self) -> list[dict]:
        """
        This function returns Mitre Att&ck data.
        """
        if not self.__attack_data:
            from pyattck import Attck
            if not self.data_path:
                self.setup_attack_cache_path()

            print(f"Loading attack data from: {self.data_path}")
            self.__attack_data = Attck(data_path=str(self.data_path))

        if self.__attack_data is None:
            raise ValueError("Attack data is not initialized. Please call generate_attack_data() first.")

    def summary_attack_data(self) -> list[dict]:
        """
        Summarize the attack data by printing relevant information.
        """
        tactics_with_techniques: list[dict] = self.__get_tactics(self.__attack_data.enterprise.tactics)

        for tac in tactics_with_techniques:
            self.logger.debug(tac['tactic_id'])
            self.logger.debug(tac['name'])
            self.logger.debug(tac['description'])
            self.logger.debug(f"Number of techniques: {len(tac['techniques'])}")
            self.logger.debug("-" * 40)

        return tactics_with_techniques

    def get_attack_data(self) -> list[dict]:
        """
        Generate synthetic attack data for testing.
        This function initializes the Attck object with the specified data path.

        Returns:
                List[dict]: A list of dictionaries containing attack data.
                                                                Keys include 'tactic_id', 'name', 'description', and 'techniques'.
                                                                'techniques' is a list of dictionaries with keys 'technique_id', 'name', and 'description'.
        Raises:
                ValueError: If the data path is not set.
        """
        if not self.data_path:
            self.setup_attack_cache_path()

        if self.data_path is None:
            raise ValueError("Data path must be set before generating attack data.")

        if self.__attack_data is None:
            self.__get_attack_data()

        return self.summary_attack_data()


if __name__ == "__main__":
    # Example usage
    utility = AttackUtility(data_path=Path("./attack_data", debug=True))
    utility.setup_attack_cache_path()
    attack_data = utility.get_attack_data()
    print(attack_data)

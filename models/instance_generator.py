import math
import random
from pathlib import Path


# ============================================================
# GENERAL PARAMETERS
# ============================================================

N_VALUES = [12, 24]
G_VALUES = [2, 3, 4]
NOISE_VALUES = [1, 5, 10]

# Directory in which the generated instances will be stored.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "sinteticas"

# Random seed used to generate the instances reported in the paper.
SEED = 12345


# ============================================================
# AUXILIARY FUNCTIONS
# ============================================================

def generate_permutation(n):
    """
    Sample uniformly at random a permutation of items 1, ..., n.
    """
    permutation = list(range(1, n + 1))
    random.shuffle(permutation)
    return permutation


def get_ranking(permutation):
    """
    Return the inverse ranking associated with a permutation.

    If item i appears in position k, ranking[i - 1] = k.
    """
    ranking = [0] * len(permutation)

    for position, item in enumerate(permutation):
        ranking[item - 1] = position

    return ranking


def generate_n1_neighbor(permutation):
    """
    Generate an N1 neighbor by swapping two consecutive items.
    """
    neighbor = permutation.copy()
    position = random.randrange(len(neighbor) - 1)

    neighbor[position], neighbor[position + 1] = (
        neighbor[position + 1],
        neighbor[position],
    )

    return neighbor


def kendall_distance(permutation_1, permutation_2):
    """
    Compute the Kendall distance between two permutations.
    """
    ranking_1 = get_ranking(permutation_1)
    ranking_2 = get_ranking(permutation_2)

    n = len(permutation_1)
    inversions = 0

    for i in range(n):
        for j in range(i + 1, n):
            if (
                ranking_1[i] < ranking_1[j]
                and ranking_2[i] > ranking_2[j]
            ) or (
                ranking_1[i] > ranking_1[j]
                and ranking_2[i] < ranking_2[j]
            ):
                inversions += 1

    return inversions


def get_component_counts(g, weight_type):
    """
    Return the number of rankings generated for each component.

    The counts always sum to 1000.

    weight_type = 1:
        Decreasing component weights.

    weight_type = 2:
        Equal or approximately equal component weights.
    """
    if g == 2:
        if weight_type == 1:
            return [667, 333]
        return [500, 500]

    if g == 3:
        if weight_type == 1:
            return [571, 286, 143]
        return [334, 333, 333]

    if g == 4:
        if weight_type == 1:
            return [533, 267, 133, 67]
        return [250, 250, 250, 250]

    raise ValueError("The number of components must be 2, 3, or 4.")


# ============================================================
# CENTRAL PERMUTATION GENERATION
# ============================================================

def generate_central_permutations(n, number_of_components):
    """
    Generate the central permutations sequentially by rejection
    sampling.

    Each candidate is sampled uniformly at random. A candidate is
    accepted only if its Kendall distance from every previously
    accepted central permutation is at least minimum_distance.

    If no candidate is accepted after max_tries attempts, the
    minimum distance is reduced by distance_step.
    """
    maximum_kendall_distance = n * (n - 1) // 2

    minimum_distance = math.ceil(
        0.20 * maximum_kendall_distance
    )

    distance_step = math.ceil(
        0.01 * maximum_kendall_distance
    )

    max_tries = 20000
    central_permutations = []

    for component in range(number_of_components):
        tries = 0

        while True:
            candidate = generate_permutation(n)

            is_valid = all(
                kendall_distance(candidate, previous)
                >= minimum_distance
                for previous in central_permutations
            )

            tries += 1

            if is_valid:
                central_permutations.append(candidate)
                break

            if tries >= max_tries:
                minimum_distance = max(
                    0,
                    minimum_distance - distance_step,
                )
                tries = 0

    return central_permutations


# ============================================================
# OUTPUT FILES
# ============================================================

def write_summary_file(
    path,
    n,
    component_counts,
    central_permutations,
    component_matrices,
):
    """
    Write a summary containing the generating permutations,
    weights, and component-specific pairwise count matrices.
    """
    total_count = sum(component_counts)
    total_value = 0.0

    with path.open(mode="w", encoding="utf-8") as file:
        file.write(f"Random seed: {SEED}\n\n")

        for component, central_permutation in enumerate(
            central_permutations
        ):
            file.write(f"COMPONENT {component + 1}\n")

            file.write("Central permutation\n")
            file.write(
                " ".join(
                    str(item)
                    for item in central_permutation
                )
            )
            file.write("\n")

            file.write("Weight\n")
            file.write(
                f"{component_counts[component] / total_count}\n"
            )

            file.write("Pairwise count matrix\n")

            for row in range(n):
                file.write(
                    " ".join(
                        f"{component_matrices[component][row][column]:4d}"
                        for column in range(n)
                    )
                )
                file.write("\n")

            component_value = 0

            for position_1 in range(n):
                for position_2 in range(position_1 + 1, n):
                    item_1 = (
                        central_permutation[position_1] - 1
                    )
                    item_2 = (
                        central_permutation[position_2] - 1
                    )

                    component_value += (
                        component_matrices[component][item_1][item_2]
                    )

            total_value += component_value

            component_objective_value = (
                n * (n - 1) / 2
                - component_value / component_counts[component]
            )

            file.write(
                "Objective value: "
                f"{component_objective_value}\n\n"
            )

        total_objective_value = (
            n * (n - 1) / 2
            - total_value / total_count
        )

        file.write(
            "TOTAL OBJECTIVE VALUE: "
            f"{total_objective_value}\n"
        )


def write_dat_file(
    path,
    n,
    number_of_components,
    number_of_swaps,
    aggregated_matrix,
):
    """
    Write the generated instance in .dat format.
    """
    with path.open(mode="w", encoding="utf-8") as file:
        file.write(f"n: {n}\n")
        file.write(f"g: {number_of_components}\n")
        file.write(f"nswaps: {number_of_swaps}\n")
        file.write("a:\n")
        file.write("[\n")

        for row in aggregated_matrix:
            file.write(
                " " + " ".join(str(value) for value in row) + "\n"
            )

        file.write("]\n")


# ============================================================
# INSTANCE GENERATION
# ============================================================

def generate_instance(
    n,
    number_of_components,
    noise_percentage,
    instance_name,
    weight_type,
    output_directory,
):
    """
    Generate one complete synthetic instance.
    """
    component_counts = get_component_counts(
        number_of_components,
        weight_type,
    )

    maximum_kendall_distance = n * (n - 1) // 2

    number_of_swaps = math.ceil(
        noise_percentage
        / 100
        * maximum_kendall_distance
    )

    central_permutations = generate_central_permutations(
        n,
        number_of_components,
    )

    # component_matrices[i][r][s] is the number of rankings
    # generated from component i in which r precedes s.
    component_matrices = [
        [
            [0 for _ in range(n)]
            for _ in range(n)
        ]
        for _ in range(number_of_components)
    ]

    for component in range(number_of_components):
        for _ in range(component_counts[component]):
            generated_ranking = (
                central_permutations[component].copy()
            )

            # Apply exactly D random adjacent transpositions.
            # Their effects may cancel, so the resulting Kendall
            # distance from the central permutation is at most D.
            for _ in range(number_of_swaps):
                generated_ranking = generate_n1_neighbor(
                    generated_ranking
                )

            # Update every pairwise comparison induced by the
            # generated complete ranking.
            for position_1 in range(n):
                for position_2 in range(position_1 + 1, n):
                    item_1 = generated_ranking[position_1] - 1
                    item_2 = generated_ranking[position_2] - 1

                    component_matrices[
                        component
                    ][item_1][item_2] += 1

    # Aggregate the component-specific pairwise count matrices.
    aggregated_matrix = [
        [
            sum(
                component_matrices[component][row][column]
                for component in range(number_of_components)
            )
            for column in range(n)
        ]
        for row in range(n)
    ]

    summary_path = (
        output_directory
        / f"Summary{instance_name}.txt"
    )

    dat_path = (
        output_directory
        / f"{instance_name}.dat"
    )

    write_summary_file(
        path=summary_path,
        n=n,
        component_counts=component_counts,
        central_permutations=central_permutations,
        component_matrices=component_matrices,
    )

    write_dat_file(
        path=dat_path,
        n=n,
        number_of_components=number_of_components,
        number_of_swaps=number_of_swaps,
        aggregated_matrix=aggregated_matrix,
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    random.seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    instance_counter = 1

    for n in N_VALUES:
        for number_of_components in G_VALUES:
            for noise_percentage in NOISE_VALUES:

                # Decreasing component weights.
                instance_name = f"R{instance_counter}"

                generate_instance(
                    n=n,
                    number_of_components=number_of_components,
                    noise_percentage=noise_percentage,
                    instance_name=instance_name,
                    weight_type=1,
                    output_directory=OUTPUT_DIR,
                )

                print(
                    f"Generated {instance_name}: "
                    f"n={n}, "
                    f"g={number_of_components}, "
                    f"p={noise_percentage}, "
                    f"weight_type=decreasing"
                )

                instance_counter += 1

                # Equal or approximately equal component weights.
                instance_name = f"R{instance_counter}"

                generate_instance(
                    n=n,
                    number_of_components=number_of_components,
                    noise_percentage=noise_percentage,
                    instance_name=instance_name,
                    weight_type=2,
                    output_directory=OUTPUT_DIR,
                )

                print(
                    f"Generated {instance_name}: "
                    f"n={n}, "
                    f"g={number_of_components}, "
                    f"p={noise_percentage}, "
                    f"weight_type=equal"
                )

                instance_counter += 1


if __name__ == "__main__":
    main()
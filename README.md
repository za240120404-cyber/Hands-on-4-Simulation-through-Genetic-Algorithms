import java.util.ArrayList;
import java.util.List;
import java.util.Random;

// =========================================================================
// 1. CLASE DATASET: Contiene los datos predefinidos del caso Benetton
// =========================================================================
class DataSet {
    private final double[] advertising; // Variable independiente (X)
    private final double[] sales;       // Variable dependiente (Y)

    public DataSet() {
        // Datos del caso Benetton (Inversión en Publicidad vs Ventas)
        this.advertising = new double[]{23.0, 26.0, 30.0, 34.0, 43.0, 48.0, 52.0, 57.0, 58.0};
        this.sales = new double[]{651.0, 762.0, 856.0, 1063.0, 1190.0, 1298.0, 1421.0, 1440.0, 1518.0};
    }

    public double[] getAdvertising() { return advertising; }
    public double[] getSales() { return sales; }
    public int size() { return advertising.length; }
}

// =========================================================================
// 2. CLASE CROMOSOMA (Individuo): Representa una solución potencial (beta_0, beta_1)
// =========================================================================
class Chromosome {
    private double beta0;
    private double beta1;
    private double fitness; // Error Cuadrático Medio (MSE) invertido o directo

    public Chromosome(double beta0, double beta1) {
        this.beta0 = beta0;
        this.beta1 = beta1;
    }

    // Evalúa qué tan buena es la solución usando el Error Cuadrático Medio (MSE)
    // Buscamos MINIMIZAR el MSE, por lo que un menor fitness real es "mejor"
    public void evaluateFitness(DataSet data) {
        double[] x = data.getAdvertising();
        double[] y = data.getSales();
        double totalError = 0;

        for (int i = 0; i < data.size(); i++) {
            double prediction = beta0 + (beta1 * x[i]);
            double error = y[i] - prediction;
            totalError += error * error;
        }
        this.fitness = totalError / data.size();
    }

    public double getBeta0() { return beta0; }
    public void setBeta0(double beta0) { this.beta0 = beta0; }
    public double getBeta1() { return beta1; }
    public void setBeta1(double beta1) { this.beta1 = beta1; }
    public double getFitness() { return fitness; }
}

// =========================================================================
// 3. CLASE ALGORITMO GENÉTICO: Controla la evolución de las soluciones
// =========================================================================
class GeneticAlgorithm {
    private final DataSet data;
    private final int populationSize;
    private final double mutationRate;
    private final int generations;
    private List<Chromosome> population;
    private final Random random;

    public GeneticAlgorithm(DataSet data, int populationSize, double mutationRate, int generations) {
        this.data = data;
        this.populationSize = populationSize;
        this.mutationRate = mutationRate;
        this.generations = generations;
        this.population = new ArrayList<>();
        this.random = new Random();
    }

    // Inicializa la población con valores aleatorios tentativos para los coeficientes
    public void initializePopulation() {
        for (int i = 0; i < populationSize; i++) {
            // Rangos iniciales lógicos estimados para el dataset de Benetton
            double randomBeta0 = random.nextDouble() * 200;       // Entre 0 y 200
            double randomBeta1 = 10 + (random.nextDouble() * 30); // Entre 10 y 40
            Chromosome chromosome = new Chromosome(randomBeta0, randomBeta1);
            chromosome.evaluateFitness(data);
            population.add(chromosome);
        }
    }

    // Ejecuta el ciclo evolutivo
    public Chromosome evolve() {
        initializePopulation();

        for (int g = 0; g < generations; g++) {
            List<Chromosome> nextGeneration = new ArrayList<>();

            // Elitismo: Pasar directamente el mejor de la generación anterior
            Chromosome best = getBestChromosome();
            nextGeneration.add(new Chromosome(best.getBeta0(), best.getBeta1()));

            while (nextGeneration.size() < populationSize) {
                // Selección por Torneo
                Chromosome parent1 = tournamentSelection();
                Chromosome parent2 = tournamentSelection();

                // Cruza (Crossover Aritmético)
                Chromosome child = crossover(parent1, parent2);

                // Mutación
                mutate(child);

                child.evaluateFitness(data);
                nextGeneration.add(child);
            }
            population = nextGeneration;
        }
        return getBestChromosome();
    }

    // Selección por torneo: Toma 3 aleatorios y se queda con el que tenga MENOR error (fitness)
    private Chromosome tournamentSelection() {
        Chromosome best = population.get(random.nextInt(populationSize));
        for (int i = 0; i < 3; i++) {
            Chromosome competitor = population.get(random.nextInt(populationSize));
            if (competitor.getFitness() < best.getFitness()) {
                best = competitor;
            }
        }
        return best;
    }

    // Cruza mediante promedio ponderado (Crossover Uniforme Continuo)
    private Chromosome crossover(Chromosome p1, Chromosome p2) {
        double alpha = random.nextDouble();
        double childBeta0 = alpha * p1.getBeta0() + (1 - alpha) * p2.getBeta0();
        double childBeta1 = alpha * p1.getBeta1() + (1 - alpha) * p2.getBeta1();
        return new Chromosome(childBeta0, childBeta1);
    }

    // Mutación Gaussiana: Añade una pequeña variación si el porcentaje lo permite
    private void mutate(Chromosome c) {
        if (random.nextDouble() < mutationRate) {
            c.setBeta0(c.getBeta0() + random.nextGaussian() * 10); // Variación para Beta0
        }
        if (random.nextDouble() < mutationRate) {
            c.setBeta1(c.getBeta1() + random.nextGaussian() * 1);  // Variación para Beta1
        }
    }

    // Busca el individuo con menor Error Cuadrático Medio
    public Chromosome getBestChromosome() {
        Chromosome best = population.get(0);
        for (Chromosome c : population) {
            if (c.getFitness() < best.getFitness()) {
                best = c;
            }
        }
        return best;
    }
}

// =========================================================================
// 4. CLASE PRINCIPAL: Coordina las especificaciones de salida solicitadas
// =========================================================================
public class Main {
    public static void main(String[] args) {
        // Carga de datos sin captura por teclado (Conforme a los specs)
        DataSet data = new DataSet();

        // Configuración de Parámetros del Algoritmo Genético
        int poblacion = 100;
        double tasaMutacion = 0.25;
        int generaciones = 2000;

        System.out.println("Ejecutando Algoritmo Genético para Regresión Lineal...");
        GeneticAlgorithm ga = new GeneticAlgorithm(data, poblacion, tasaMutacion, generaciones);
        
        // Ejecución de la optimización
        Chromosome mejorSolucion = ga.evolve();

        double b0 = mejorSolucion.getBeta0();
        double b1 = mejorSolucion.getBeta1();

        // -----------------------------------------------------------------
        // OUTPUT 1: Imprimir Ecuación de Regresión Lineal Simple sustituyendo valores
        // -----------------------------------------------------------------
        System.out.println("\n=========================================================");
        System.out.println("1. ECUACIÓN DE REGRESIÓN LINEAL OPTIMIZADA");
        System.out.println("=========================================================");
        System.out.printf("Sales = %.4f + (%.4f * Advertising)%n", b0, b1);
        System.out.printf("[Error Cuadrático Medio (MSE) final: %.4f]%n", mejorSolucion.getFitness());

        // -----------------------------------------------------------------
        // OUTPUT 2: Simular, al menos 5 experimentos con valores desconocidos
        // -----------------------------------------------------------------
        System.out.println("\n=========================================================");
        System.out.println("2. SIMULACIÓN DE EXPERIMENTOS (Valores Desconocidos)");
        System.out.println("=========================================================");
        
        // 5 Valores de simulación para simular escenarios fuera del dataset original
        double[] nuevosValoresAdvertising = {15.0, 38.0, 45.5, 65.0, 72.0};

        for (int i = 0; i < nuevosValoresAdvertising.length; i++) {
            double adv = nuevosValoresAdvertising[i];
            // Aplicación de la fórmula matemática obtenida
            double prediccionVentas = b0 + (b1 * adv);
            
            System.out.printf("Experimento %d | Inversión (Advertising): %5.1f -> Ventas Estimadas (Sales): %8.2f%n", 
                    (i + 1), adv, prediccionVentas);
        }
        System.out.println("=========================================================");
    }
}

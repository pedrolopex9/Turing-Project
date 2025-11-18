import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;

import java.io.*;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.Map;

public class Server {

    public static void main(String[] args) throws IOException {
        int port = 8080;
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        
        server.createContext("/", new StaticHandler());
        server.createContext("/run", new SimulationHandler());

        server.setExecutor(null);
        System.out.println(">>> [JAVA] SERVIDOR PRONTO EM: http://localhost:" + port);
        server.start();
    }

    static class StaticHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange t) throws IOException {
            String path = "index.html";
            File file = new File(path);
            if (file.exists()) {
                t.getResponseHeaders().set("Content-Type", "text/html; charset=UTF-8");
                t.sendResponseHeaders(200, file.length());
                OutputStream os = t.getResponseBody();
                Files.copy(file.toPath(), os);
                os.close();
            } else {
                String resp = "ERRO: index.html nao encontrado. Verifique a pasta!";
                t.sendResponseHeaders(404, resp.length());
                t.getResponseBody().write(resp.getBytes());
                t.getResponseBody().close();
            }
        }
    }

    static class SimulationHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange t) throws IOException {
            String query = t.getRequestURI().getQuery();
            Map<String, String> params = queryToMap(query);
            
            String exe = params.getOrDefault("exe", "modelo10.exe");
            String method = params.getOrDefault("method", "pso");
            String goal = params.getOrDefault("goal", "min");
            String dim = params.getOrDefault("dim", "9");

            System.out.println("\n>>> [JAVA] INICIANDO: " + method.toUpperCase() + " -> " + exe);

            // --- CORRECAO CRITICA: FORCAR UTF-8 ---
            // Removi acentos dos comentarios para evitar erro no javac
            ProcessBuilder pb = new ProcessBuilder(
                "python", "-u", "universal_worker.py",
                "--exe", exe,
                "--method", method,
                "--goal", goal,
                "--dim", dim,
                "--pop", "20",
                "--iter", "50"
            );
            
            // Isso impede que o Python crashe por causa de acentos no Windows
            pb.environment().put("PYTHONIOENCODING", "utf-8");
            pb.redirectErrorStream(true); 

            Process process = pb.start();

            t.getResponseHeaders().set("Content-Type", "text/plain; charset=UTF-8");
            t.sendResponseHeaders(200, 0);

            // Le a saida usando UTF-8 explicitamente
            BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), "UTF-8")
            );
            OutputStream os = t.getResponseBody();
            
            String line;
            try {
                while ((line = reader.readLine()) != null) {
                    // Mostra no terminal do VS Code
                    System.out.println("[PYTHON]: " + line);
                    os.write((line + "\n").getBytes("UTF-8"));
                    os.flush(); 
                }
            } catch (Exception e) {
                System.out.println(">>> [JAVA] Erro de conexao: " + e.getMessage());
            } finally {
                os.close();
                process.destroy();
            }
        }

        private Map<String, String> queryToMap(String query) {
            Map<String, String> result = new HashMap<>();
            if (query == null) return result;
            for (String param : query.split("&")) {
                String[] entry = param.split("=");
                if (entry.length > 1) result.put(entry[0], entry[1]);
            }
            return result;
        }
    }
}